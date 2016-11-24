/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package wekaclustering;

import java.util.HashMap;
import java.util.Random;
import weka.classifiers.rules.DecisionTableHashKey;
import weka.clusterers.NumberOfClustersRequestable;
import weka.clusterers.RandomizableClusterer;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.Instances;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformationHandler;
import weka.core.WeightedInstancesHandler;
import weka.core.DistanceFunction;
import weka.core.EuclideanDistance;
import weka.core.Instance;
import weka.core.Utils;
import static weka.core.pmml.PMMLUtils.pad;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;

/**
 *
 * @author Asus
 */
public class myKMeans extends RandomizableClusterer implements NumberOfClustersRequestable, WeightedInstancesHandler, TechnicalInformationHandler {
    private int m_maxIterations = 100;
    private int m_iterations = 0;
    private DistanceFunction m_distanceFunction = new EuclideanDistance();
    private Instances clusterCentroids;
    private int[] m_clusterSizes;
    
    /**
    * number of clusters to generate
    */
    private int m_numClusters = 2;
    
    /**
    * replace missing values in training instances
    */
    private ReplaceMissingValues m_replaceMissingFilter;
    
    /**
    * Replace missing values globally?
    */
    private boolean m_dontReplaceMissing = false;
    
    /**
    * Preserve order of instances
    */
    private boolean m_preserveOrder = false;
    
    /**
    * Assignments obtained
    */
    protected int[] m_assignments = null;
    
    /**
    * For each cluster, holds the frequency counts for the values of each nominal
    * attribute
    */
    private int[][][] m_clusterNominalCounts;
    private int[][] m_clusterMissingCounts;
    
    /**
    * Stats on the full data set for comparison purposes In case the attribute is
    * numeric the value is the mean if is being used the Euclidian distance or
    * the median if Manhattan distance and if the attribute is nominal then it's
    * mode is saved
    */
    private double[] m_fullMeansOrModes;
    private int[][] m_fullNominalCounts;
    private int[] m_fullMissingCounts;
    
    
    /**
     * Holds the squared errors for all clusters
     */
    private double[] squaredErrors;
    
    /**
    * Returns default capabilities of the clusterer.
    * 
    * @return the capabilities of this clusterer
    */
    @Override
    public Capabilities getCapabilities() {
        Capabilities result = super.getCapabilities();
        result.disableAll();
        result.enable(Capability.NO_CLASS);

        // attributes
        result.enable(Capability.NOMINAL_ATTRIBUTES);
        result.enable(Capability.NUMERIC_ATTRIBUTES);
        result.enable(Capability.MISSING_VALUES);

        return result;
    }
    
    /**
    * Generates a clusterer. Has to initialize all fields of the clusterer that
    * are not being set via options.
    * 
    * @param data set of instances serving as training data
    * @throws Exception if the clusterer has not been generated successfully
    */
    @Override
    public void buildClusterer(Instances data) throws Exception {
        // can clusterer handle the data?
        getCapabilities().testWithFail(data);
        m_iterations = 0;
        m_replaceMissingFilter = new ReplaceMissingValues();
        Instances instances = new Instances(data);
        
        instances.setClassIndex(-1);
        if (!m_dontReplaceMissing) {
            m_replaceMissingFilter.setInputFormat(instances);
            instances = Filter.useFilter(instances, m_replaceMissingFilter);
        }
        
        m_fullMissingCounts = new int[instances.numAttributes()];
        m_fullNominalCounts = new int[instances.numAttributes()][0];
        m_fullMeansOrModes = moveCentroid(0, instances, false);
        
        for(int i=0; i<instances.numAttributes(); i++) {
            m_fullMissingCounts[i] = instances.attributeStats(i).missingCount;
            if(instances.attribute(i).isNumeric()) {
                if(m_fullMissingCounts[i] == instances.numInstances()) {
                    m_fullMeansOrModes[i] = Double.NaN;
                }
            } 
            else {
                m_fullNominalCounts[i] = instances.attributeStats(i).nominalCounts;
                if(m_fullMissingCounts[i] > m_fullNominalCounts[i][Utils.maxIndex(m_fullNominalCounts[i])]) {
                    m_fullMeansOrModes[i] = -1;
                }
            }
        }
        
        clusterCentroids = new Instances(instances, m_numClusters);
        int[] clusterAssignments = new int[instances.numInstances()];
        
        if(m_preserveOrder) {
            m_assignments = clusterAssignments;
        }
        
        m_distanceFunction.setInstances(instances);
        
        Instances initInstances = null;
        if(m_preserveOrder) {
            initInstances = new Instances(instances);
        }
        else {
            initInstances = instances;
        }
        
        initCentroids(initInstances);
        
        m_numClusters = clusterCentroids.numInstances();
        
        // removing reference
        initInstances = null;
        
        int i;
        boolean converged = false;
        int emptyClusterCount;
        Instances[] tempClusterInstances = new Instances[m_numClusters];
        squaredErrors = new double[m_numClusters];
        m_clusterNominalCounts = new int[m_numClusters][instances.numAttributes()][0];
        m_clusterMissingCounts = new int[m_numClusters][instances.numAttributes()];
        while (!converged) {
            emptyClusterCount = 0;
            m_iterations++;
            converged = true;
            
            // Clusters Instances
            for (i = 0; i < instances.numInstances(); i++) {
                Instance toCluster = instances.instance(i);
                int newCluster = clusterProcessedInstance(toCluster, true);
                if (newCluster != clusterAssignments[i]) {
                    converged = false;
                }
                clusterAssignments[i] = newCluster;
            }

            // update centroids
            clusterCentroids = new Instances(instances, m_numClusters);
            for (i = 0; i < m_numClusters; i++) {
                tempClusterInstances[i] = new Instances(instances, 0);
            }
            for (i = 0; i < instances.numInstances(); i++) {
                tempClusterInstances[clusterAssignments[i]].add(instances.instance(i));
            }
            for (i = 0; i < m_numClusters; i++) {
                if (tempClusterInstances[i].numInstances() == 0) {
                    // empty cluster
                    emptyClusterCount++;
                } else {
                    moveCentroid(i, tempClusterInstances[i], true);
                }
            }

            if (m_iterations == m_maxIterations) {
                converged = true;
            }

            if (emptyClusterCount > 0) {
                m_numClusters -= emptyClusterCount;
                if (converged) {
                    Instances[] t = new Instances[m_numClusters];
                    int index = 0;
                    for (int k = 0; k < tempClusterInstances.length; k++) {
                        if (tempClusterInstances[k].numInstances() > 0) {
                            t[index] = tempClusterInstances[k];

                            for (i = 0; i < tempClusterInstances[k].numAttributes(); i++) {
                                m_clusterNominalCounts[index][i] = m_clusterNominalCounts[k][i];
                            }
                            index++;
                        }
                    }
                    tempClusterInstances = t;
                } else {
                    tempClusterInstances = new Instances[m_numClusters];
                }
            }

            if (!converged) {
                squaredErrors = new double[m_numClusters];
                m_clusterNominalCounts = new int[m_numClusters][instances.numAttributes()][0];
            }
        }

        m_clusterSizes = new int[m_numClusters];
        for (i = 0; i < m_numClusters; i++) {
            m_clusterSizes[i] = tempClusterInstances[i].numInstances();
        }

        // Save memory!!
        m_distanceFunction.clean();
    }
    
    /**
    * clusters an instance that has been through the filters
    * 
    * @param instance the instance to assign a cluster to
    * @param updateErrors if true, update the within clusters sum of errors
    * @return a cluster number
    */
    private int clusterProcessedInstance(Instance instance, boolean updateErrors) {
        double minDist = Integer.MAX_VALUE;
        int bestCluster = 0;
        for (int i = 0; i < m_numClusters; i++) {
            double dist = m_distanceFunction.distance(instance, clusterCentroids.instance(i));
            if (dist < minDist) {
                minDist = dist;
                bestCluster = i;
            }
        }
        if (updateErrors) {
            if (m_distanceFunction instanceof EuclideanDistance) {
                // Euclidean distance to Squared Euclidean distance
                minDist *= minDist;
            }
            squaredErrors[bestCluster] += minDist;
        }
        return bestCluster;
    }
    
    private void initCentroids(Instances data) throws Exception {
        Random RandomO = new Random(getSeed());
        int instIndex;
        HashMap initCluster = new HashMap();
        DecisionTableHashKey hk = null;

        // Pick Random Centroids
        for (int j = data.numInstances() - 1; j >= 0; j--) {
            instIndex = RandomO.nextInt(j + 1);
            hk = new DecisionTableHashKey(data.instance(instIndex), data.numAttributes(), true);
            if (!initCluster.containsKey(hk)) {
                clusterCentroids.add(data.instance(instIndex));
                initCluster.put(hk, null);
            }
            data.swap(j, instIndex);

            if (clusterCentroids.numInstances() == m_numClusters) {
                break;
            }
        }
    }
    
    /**
    * Classifies a given instance.
    * 
    * @param instance the instance to be assigned to a cluster
    * @return the number of the assigned cluster as an interger if the class is
    *         enumerated, otherwise the predicted value
    * @throws Exception if instance could not be classified successfully
    */
    @Override
    public int clusterInstance(Instance instance) throws Exception {
        Instance inst = null;
        if (!m_dontReplaceMissing) {
            m_replaceMissingFilter.input(instance);
            m_replaceMissingFilter.batchFinished();
            inst = m_replaceMissingFilter.output();
        } else {
            inst = instance;
        }

        return clusterProcessedInstance(inst, false);
    }

    @Override
    public int numberOfClusters() throws Exception {
        return m_numClusters;
    }

    @Override
    public void setNumClusters(int i) throws Exception {
        if (i <= 0) {
            throw new Exception("Number of clusters must be > 0");
        }
        m_numClusters = i;
    }

    @Override
    public TechnicalInformation getTechnicalInformation() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }
    
    /**
    * Move the centroid to it's new coordinates. Generate the centroid
    * coordinates based on it's members (objects assigned to the cluster of the
    * centroid) and the distance function being used.
    * 
    * @param centroidIndex index of the centroid which the coordinates will be
    *          computed
    * @param members the objects that are assigned to the cluster of this
    *          centroid
    * @param updateClusterInfo if the method is supposed to update the m_Cluster
    *          arrays
    * @return the centroid coordinates
    */
    protected double[] moveCentroid(int centroidIndex, Instances members, boolean updateClusterInfo) {
        double[] vals = new double[members.numAttributes()];

        for (int j = 0; j < members.numAttributes(); j++) {

            // in case of Euclidian distance the centroid is the mean point
            // in both cases, if the attribute is nominal, the centroid is the mode
            if (m_distanceFunction instanceof EuclideanDistance || members.attribute(j).isNominal()) {
                vals[j] = members.meanOrMode(j);
            }

            if (updateClusterInfo) {
                m_clusterMissingCounts[centroidIndex][j] = members.attributeStats(j).missingCount;
                m_clusterNominalCounts[centroidIndex][j] = members.attributeStats(j).nominalCounts;
                if (members.attribute(j).isNominal()) {
                    if (m_clusterMissingCounts[centroidIndex][j] > m_clusterNominalCounts[centroidIndex][j][Utils.maxIndex(m_clusterNominalCounts[centroidIndex][j])]) {
                        vals[j] = Instance.missingValue(); // mark mode as missing
                    }
                } else {
                    if (m_clusterMissingCounts[centroidIndex][j] == members.numInstances()) {
                       vals[j] = Instance.missingValue(); // mark mean as missing
                    }
                }
            }
        }
        if (updateClusterInfo) {
          clusterCentroids.add(new Instance(1.0, vals));
        }
        return vals;
    }
    
    /**
    * return a string describing this clusterer
    * 
    * @return a description of the clusterer as a string
    */
    @Override
    public String toString() {
        if (clusterCentroids == null) {
            return "No clusterer built yet!";
        }

        int maxWidth = 0;
        int maxAttWidth = 0;
        boolean containsNumeric = false;
        for (int i = 0; i < m_numClusters; i++) {
            for (int j = 0; j < clusterCentroids.numAttributes(); j++) {
                if (clusterCentroids.attribute(j).name().length() > maxAttWidth) {
                    maxAttWidth = clusterCentroids.attribute(j).name().length();
                }
                if (clusterCentroids.attribute(j).isNumeric()) {
                    containsNumeric = true;
                    double width = Math.log(Math.abs(clusterCentroids.instance(i).value(j))) / Math.log(10.0);
                    if (width < 0) {
                        width = 1;
                    }
                    // decimal + # decimal places + 1
                    width += 6.0;
                    if ((int) width > maxWidth) {
                        maxWidth = (int) width;
                    }
                }
            }
        }

        for (int i = 0; i < clusterCentroids.numAttributes(); i++) {
            if (clusterCentroids.attribute(i).isNominal()) {
                Attribute a = clusterCentroids.attribute(i);
                for (int j = 0; j < clusterCentroids.numInstances(); j++) {
                    String val = a.value((int) clusterCentroids.instance(j).value(i));
                    if (val.length() > maxWidth) {
                        maxWidth = val.length();
                    }
                }
                for (int j = 0; j < a.numValues(); j++) {
                    String val = a.value(j) + " ";
                    if (val.length() > maxAttWidth) {
                        maxAttWidth = val.length();
                    }
                }
            }
        }

        // check for size of cluster sizes
        for (int m_ClusterSize : m_clusterSizes) {
            String size = "(" + m_ClusterSize + ")";
            if (size.length() > maxWidth) {
                maxWidth = size.length();
            }
        }

        if (maxAttWidth < "missing".length()) {
            maxAttWidth = "missing".length();
        }

        String plusMinus = "+/-";
        maxAttWidth += 2;
        if (containsNumeric) {
            maxWidth += plusMinus.length();
        }
        if (maxAttWidth < "Attribute".length() + 2) {
            maxAttWidth = "Attribute".length() + 2;
        }

        if (maxWidth < "Full Data".length()) {
            maxWidth = "Full Data".length() + 1;
        }

        if (maxWidth < "missing".length()) {
            maxWidth = "missing".length() + 1;
        }

        StringBuffer temp = new StringBuffer();

        temp.append("\nkMeans\n======\n");
        temp.append("\nNumber of iterations: " + m_iterations + "\n");

        if (m_distanceFunction instanceof EuclideanDistance) {
            temp.append("Within cluster sum of squared errors: " + Utils.sum(squaredErrors));
        }

        if (!m_dontReplaceMissing) {
            temp.append("\nMissing values globally replaced with mean/mode");
        }

        temp.append("\n\nCluster centroids:\n");
        temp.append(pad("Cluster#", " ", (maxAttWidth + (maxWidth * 2 + 2)) - "Cluster#".length(), true));

        temp.append("\n");
        temp.append(pad("Attribute", " ", maxAttWidth - "Attribute".length(), false));

        temp.append(pad("Full Data", " ", maxWidth + 1 - "Full Data".length(), true));

        // cluster numbers
        for (int i = 0; i < m_numClusters; i++) {
            String clustNum = "" + i;
            temp.append(pad(clustNum, " ", maxWidth + 1 - clustNum.length(), true));
        }
        temp.append("\n");

        // cluster sizes
        String cSize = "(" + Utils.sum(m_clusterSizes) + ")";
        temp.append(pad(cSize, " ", maxAttWidth + maxWidth + 1 - cSize.length(),true));
        for (int i = 0; i < m_numClusters; i++) {
            cSize = "(" + m_clusterSizes[i] + ")";
            temp.append(pad(cSize, " ", maxWidth + 1 - cSize.length(), true));
        }
        temp.append("\n");

        temp.append(pad("", "=", maxAttWidth + (maxWidth * (clusterCentroids.numInstances() + 1) + clusterCentroids.numInstances() + 1), true));
        temp.append("\n");

        for (int i = 0; i < clusterCentroids.numAttributes(); i++) {
            String attName = clusterCentroids.attribute(i).name();
            temp.append(attName);
            for (int j = 0; j < maxAttWidth - attName.length(); j++) {
                temp.append(" ");
            }

            String strVal;
            String valMeanMode;
            
            // full data
            if (clusterCentroids.attribute(i).isNominal()) {
                if (m_fullMeansOrModes[i] == -1) { // missing
                    valMeanMode = pad("missing", " ", maxWidth + 1 - "missing".length(), true);
                } else {
                    valMeanMode = pad((strVal = clusterCentroids.attribute(i).value((int) m_fullMeansOrModes[i])), " ", maxWidth + 1 - strVal.length(), true);
                }
            } else {
                if (Double.isNaN(m_fullMeansOrModes[i])) {
                    valMeanMode = pad("missing", " ", maxWidth + 1 - "missing".length(), true);
                } else {
                    valMeanMode = pad((strVal = Utils.doubleToString(m_fullMeansOrModes[i], maxWidth, 4).trim()), " ", maxWidth + 1 - strVal.length(), true);
                }
            }
            temp.append(valMeanMode);

            for (int j = 0; j < m_numClusters; j++) {
                if (clusterCentroids.attribute(i).isNominal()) {
                    if (clusterCentroids.instance(j).isMissing(i)) {
                        valMeanMode = pad("missing", " ", maxWidth + 1 - "missing".length(), true);
                    } else {
                        valMeanMode = pad((strVal = clusterCentroids.attribute(i).value((int) clusterCentroids.instance(j).value(i))), " ", maxWidth + 1 - strVal.length(), true);
                    }
                } else {
                    if (clusterCentroids.instance(j).isMissing(i)) {
                      valMeanMode = pad("missing", " ", maxWidth + 1 - "missing".length(), true);
                    } else {
                        valMeanMode = pad((strVal = Utils.doubleToString(clusterCentroids.instance(j).value(i), maxWidth, 4).trim()), " ", maxWidth + 1 - strVal.length(), true);
                    }
                }
                temp.append(valMeanMode);
            }
            temp.append("\n");
        }

        temp.append("\n\n");
        return temp.toString();
    }
    
    /**
    * set the maximum number of iterations to be executed
    * 
    * @param n the maximum number of iterations
    * @throws Exception if maximum number of iteration is smaller than 1
    */
    public void setMaxIterations(int n) throws Exception {
        if (n <= 0) {
          throw new Exception("Maximum number of iterations must be > 0");
        }
        m_maxIterations = n;
    }
    
    /**
    * sets the distance function to use for instance comparison.
    * 
    * @param df the new distance function to use
    * @throws Exception if instances cannot be processed
    */
    public void setDistanceFunction(DistanceFunction df) throws Exception {
        if (!(df instanceof EuclideanDistance)) {
          throw new Exception(
            "MyKMeans currently only supports the Euclidean distance.");
        }
        m_distanceFunction = df;
    }
    
    @Override
    public void setOptions(String[] options) throws Exception {
        m_dontReplaceMissing = Utils.getFlag("M", options);

        String optionString = Utils.getOption('N', options);

        if (optionString.length() != 0) {
            setNumClusters(Integer.parseInt(optionString));
        }

        optionString = Utils.getOption("I", options);
        if (optionString.length() != 0) {
            setMaxIterations(Integer.parseInt(optionString));
        }

        String distFunctionClass = Utils.getOption('A', options);
        if (distFunctionClass.length() != 0) {
            String distFunctionClassSpec[] = Utils.splitOptions(distFunctionClass);
            if (distFunctionClassSpec.length == 0) {
                throw new Exception("Invalid DistanceFunction specification string.");
            }
            String className = distFunctionClassSpec[0];
            distFunctionClassSpec[0] = "";

            setDistanceFunction((DistanceFunction) Utils.forName(DistanceFunction.class, className, distFunctionClassSpec));
        } else {
            setDistanceFunction(new EuclideanDistance());
        }

        m_preserveOrder = Utils.getFlag("O", options);

        super.setOptions(options);
    }
    
    public myKMeans() {
        
    }
}
