/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package wekaclustering;

import weka.clusterers.NumberOfClustersRequestable;
import weka.clusterers.RandomizableClusterer;
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

/**
 *
 * @author Asus
 */
public class myKMeans extends RandomizableClusterer implements NumberOfClustersRequestable, WeightedInstancesHandler, TechnicalInformationHandler {
    private int m_maxIterations = 100;
    private int m_iterations = 0;
    private DistanceFunction m_distanceFunction = new EuclideanDistance();
    private Instances m_clusterCentroids;
    private int[] m_clusterSizes;
    
    /**
    * For each cluster, holds the frequency counts for the values of each nominal
    * attribute
    */
    private int[][][] m_clusterNominalCounts;
    private int[][] m_clusterMissingCounts;
    
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
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public int numberOfClusters() throws Exception {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public void setNumClusters(int i) throws Exception {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
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
    protected double[] moveCentroid(int centroidIndex, Instances members,
        boolean updateClusterInfo) {
        double[] vals = new double[members.numAttributes()];

        for (int j = 0; j < members.numAttributes(); j++) {

          // in case of Euclidian distance the centroid is the mean point
          // in case of Manhattan distance the centroid is the median point
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
              if (m_clusterMissingCounts[centroidIndex][j] == members
                .numInstances()) {
                vals[j] = Instance.missingValue(); // mark mean as missing
              }
            }
          }
        }
        if (updateClusterInfo) {
          m_clusterCentroids.add(new Instance(1.0, vals));
        }
        return vals;
    }
    
    public myKMeans() {
        
    }
}
