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
    private int maxIterations = 100;
    private int iterations = 0;
    private DistanceFunction distanceFunction = new EuclideanDistance();
    private Instances clusterCentroids;
    private int[] clusterSizes;
    
    private int clustersCount = 2;
    
    private ReplaceMissingValues m_replaceMissingFilter;
    
    private boolean replaceMissing = true;
    
    private boolean preserverOrder = false;
    
    private int[][][] clusterNominalCounts;
    private int[][] clusterMissingCounts;
    
    private double[] fullMeansOrModes;
    private int[][] fullNominalCounts;
    private int[] fullMissingCounts;
    
    private double[] squaredErrors;
    
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
    
    @Override
    public void buildClusterer(Instances data) throws Exception {
        boolean converged = false;
        int emptyClusterCount;
        Instances[] tempClusterInstances = new Instances[clustersCount];

        // can clusterer handle the data?
        getCapabilities().testWithFail(data);
        iterations = 0;
        m_replaceMissingFilter = new ReplaceMissingValues();
        Instances instances = new Instances(data);
        
        instances.setClassIndex(-1);
        if (replaceMissing) {
            m_replaceMissingFilter.setInputFormat(instances);
            instances = Filter.useFilter(instances, m_replaceMissingFilter);
        }
        
        fullMissingCounts = new int[instances.numAttributes()];
        fullNominalCounts = new int[instances.numAttributes()][0];
        fullMeansOrModes = moveCentroid(0, instances, false);
        
        for(int i=0; i<instances.numAttributes(); i++) {
            fullMissingCounts[i] = instances.attributeStats(i).missingCount;
            if(instances.attribute(i).isNumeric()) {
                if(fullMissingCounts[i] == instances.numInstances()) {
                    fullMeansOrModes[i] = Double.NaN;
                }
            } 
            else {
                fullNominalCounts[i] = instances.attributeStats(i).nominalCounts;
                if(fullMissingCounts[i] > fullNominalCounts[i][Utils.maxIndex(fullNominalCounts[i])]) {
                    fullMeansOrModes[i] = -1;
                }
            }
        }
        
        clusterCentroids = new Instances(instances, clustersCount);
        int[] clusterAssignments = new int[instances.numInstances()];
        
        distanceFunction.setInstances(instances);
        
        Instances initInstances = null;
        if(preserverOrder) {
            initInstances = new Instances(instances);
        }
        else {
            initInstances = instances;
        }
        
        initCentroids(initInstances);
        
        clustersCount = clusterCentroids.numInstances();
        
        // removing reference
        initInstances = null;

        squaredErrors = new double[clustersCount];
        clusterNominalCounts = new int[clustersCount][instances.numAttributes()][0];
        clusterMissingCounts = new int[clustersCount][instances.numAttributes()];
        while (!converged) {
            emptyClusterCount = 0;
            iterations++;
            converged = true;
            
            // Initialize temporary cluster
            for (int i = 0; i < clustersCount; i++) {
                tempClusterInstances[i] = new Instances(instances, 0);
            }
            
            // Clusters Instances
            for (int i = 0; i < instances.numInstances(); i++) {
                Instance toCluster = instances.instance(i);
                int newCluster = clusterProcessedInstance(toCluster, true);
                if (newCluster != clusterAssignments[i]) {
                    converged = false;
                }
                clusterAssignments[i] = newCluster;
                tempClusterInstances[clusterAssignments[i]].add(instances.instance(i));
            }

            // Update centroids
            clusterCentroids = new Instances(instances, clustersCount);
            for (int i = 0; i < clustersCount; i++) {
                if (tempClusterInstances[i].numInstances() == 0) {
                    // empty cluster
                    emptyClusterCount++;
                } else {
                    double[] attributes = moveCentroid(i, tempClusterInstances[i], true);
                    clusterCentroids.add(new Instance(1.0, attributes));
                }
            }

            if (iterations == maxIterations) {
                converged = true;
            }

            if (emptyClusterCount > 0) {
                clustersCount -= emptyClusterCount;
                if (converged) {
                    Instances[] t = new Instances[clustersCount];
                    int index = 0;
                    for (int k = 0; k < tempClusterInstances.length; k++) {
                        if (tempClusterInstances[k].numInstances() > 0) {
                            t[index] = tempClusterInstances[k];

                            for (int i = 0; i < tempClusterInstances[k].numAttributes(); i++) {
                                clusterNominalCounts[index][i] = clusterNominalCounts[k][i];
                            }
                            index++;
                        }
                    }
                    tempClusterInstances = t;
                } else {
                    tempClusterInstances = new Instances[clustersCount];
                }
            }

            if (!converged) {
                squaredErrors = new double[clustersCount];
                clusterNominalCounts = new int[clustersCount][instances.numAttributes()][0];
            }
        }

        clusterSizes = new int[clustersCount];
        for (int i = 0; i < clustersCount; i++) {
            clusterSizes[i] = tempClusterInstances[i].numInstances();
        }

        // Save memory!!
        distanceFunction.clean();
    }
    
    private int clusterProcessedInstance(Instance instance, boolean updateErrors) {
        double minDist = Integer.MAX_VALUE;
        int bestCluster = 0;
        for (int i = 0; i < clustersCount; i++) {
            double dist = distanceFunction.distance(instance, clusterCentroids.instance(i));
            if (dist < minDist) {
                minDist = dist;
                bestCluster = i;
            }
        }
        if (updateErrors) {
            if (distanceFunction instanceof EuclideanDistance) {
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

        // Pick Random Centroids
        for (int j = data.numInstances() - 1; j >= 0; j--) {
            instIndex = RandomO.nextInt(j + 1);
            clusterCentroids.add(data.instance(instIndex));
            data.swap(j, instIndex);

            if (clusterCentroids.numInstances() == clustersCount) {
                break;
            }
        }
    }
    
    @Override
    public int clusterInstance(Instance instance) throws Exception {
        Instance inst = null;
        if (replaceMissing) {
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
        return clustersCount;
    }

    @Override
    public void setNumClusters(int i) throws Exception {
        if (i <= 0) {
            throw new Exception("Number of clusters must be > 0");
        }
        clustersCount = i;
    }

    @Override
    public TechnicalInformation getTechnicalInformation() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }
    
    protected double[] moveCentroid(int centroidIndex, Instances members, boolean updateClusterInfo) {
        double[] vals = new double[members.numAttributes()];

        for (int j = 0; j < members.numAttributes(); j++) {

            // in case of Euclidian distance the centroid is the mean point
            // in both cases, if the attribute is nominal, the centroid is the mode
            if (distanceFunction instanceof EuclideanDistance || members.attribute(j).isNominal()) {
                vals[j] = members.meanOrMode(j);
            }

            if (updateClusterInfo) {
                clusterMissingCounts[centroidIndex][j] = members.attributeStats(j).missingCount;
                clusterNominalCounts[centroidIndex][j] = members.attributeStats(j).nominalCounts;
                if (members.attribute(j).isNominal()) {
                    if (clusterMissingCounts[centroidIndex][j] > clusterNominalCounts[centroidIndex][j][Utils.maxIndex(clusterNominalCounts[centroidIndex][j])]) {
                        vals[j] = Instance.missingValue(); // mark mode as missing
                    }
                } else {
                    if (clusterMissingCounts[centroidIndex][j] == members.numInstances()) {
                       vals[j] = Instance.missingValue(); // mark mean as missing
                    }
                }
            }
        }

        return vals;
    }
    
    @Override
    public String toString() {
        if (clusterCentroids == null) {
            return "No clusterer built yet!";
        }

        int maxWidth = 0;
        int maxAttWidth = 0;
        boolean containsNumeric = false;
        for (int i = 0; i < clustersCount; i++) {
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
        for (int m_ClusterSize : clusterSizes) {
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
        temp.append("\nNumber of iterations: " + iterations + "\n");

        if (distanceFunction instanceof EuclideanDistance) {
            temp.append("Within cluster sum of squared errors: " + Utils.sum(squaredErrors));
        }

        if (replaceMissing) {
            temp.append("\nMissing values globally replaced with mean/mode");
        }

        temp.append("\n\nCluster centroids:\n");
        temp.append(pad("Cluster#", " ", (maxAttWidth + (maxWidth * 2 + 2)) - "Cluster#".length(), true));

        temp.append("\n");
        temp.append(pad("Attribute", " ", maxAttWidth - "Attribute".length(), false));

        temp.append(pad("Full Data", " ", maxWidth + 1 - "Full Data".length(), true));

        // cluster numbers
        for (int i = 0; i < clustersCount; i++) {
            String clustNum = "" + i;
            temp.append(pad(clustNum, " ", maxWidth + 1 - clustNum.length(), true));
        }
        temp.append("\n");

        // cluster sizes
        String cSize = "(" + Utils.sum(clusterSizes) + ")";
        temp.append(pad(cSize, " ", maxAttWidth + maxWidth + 1 - cSize.length(),true));
        for (int i = 0; i < clustersCount; i++) {
            cSize = "(" + clusterSizes[i] + ")";
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
                if (fullMeansOrModes[i] == -1) { // missing
                    valMeanMode = pad("missing", " ", maxWidth + 1 - "missing".length(), true);
                } else {
                    valMeanMode = pad((strVal = clusterCentroids.attribute(i).value((int) fullMeansOrModes[i])), " ", maxWidth + 1 - strVal.length(), true);
                }
            } else {
                if (Double.isNaN(fullMeansOrModes[i])) {
                    valMeanMode = pad("missing", " ", maxWidth + 1 - "missing".length(), true);
                } else {
                    valMeanMode = pad((strVal = Utils.doubleToString(fullMeansOrModes[i], maxWidth, 4).trim()), " ", maxWidth + 1 - strVal.length(), true);
                }
            }
            temp.append(valMeanMode);

            for (int j = 0; j < clustersCount; j++) {
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
    
    public void setMaxIterations(int n) throws Exception {
        if (n <= 0) {
          throw new Exception("Maximum number of iterations must be > 0");
        }
        maxIterations = n;
    }
    
    public void setDistanceFunction(DistanceFunction df) throws Exception {
        if (!(df instanceof EuclideanDistance)) {
          throw new Exception(
            "MyKMeans currently only supports the Euclidean distance.");
        }
        distanceFunction = df;
    }
    
    @Override
    public void setOptions(String[] options) throws Exception {
        replaceMissing = Utils.getFlag("M", options);

        String optionString = Utils.getOption('N', options);

        if (optionString.length() != 0) {
            setNumClusters(Integer.parseInt(optionString));
        }

        optionString = Utils.getOption("I", options);
        if (optionString.length() != 0) {
            setMaxIterations(Integer.parseInt(optionString));
        }

        preserverOrder = Utils.getFlag("O", options);

        super.setOptions(options);
    }
    
    public myKMeans() {
        
    }
}
