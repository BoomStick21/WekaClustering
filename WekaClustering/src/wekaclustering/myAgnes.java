/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package wekaclustering;

import java.util.ArrayList;
import java.util.Enumeration;
import weka.clusterers.AbstractClusterer;
import weka.core.DistanceFunction;
import weka.core.EuclideanDistance;
import weka.core.Instance;
import weka.core.Instances;


public class myAgnes extends AbstractClusterer {
    private ArrayList<ArrayList<Double>> distanceMatrix;
    private ArrayList<Cluster> clusters;
    private int numClusters;
    private int type;
    
    protected DistanceFunction distFunc = new EuclideanDistance();
    private final int SINGLE = 0;
    private final int COMPLETE = 1;
    
    public myAgnes(int _numClusters, int _type) {
        super();
        numClusters = _numClusters;
        type = _type;
        distanceMatrix = new ArrayList<ArrayList<Double>>();
        clusters = new ArrayList<Cluster>();
    }
    
    public void mergeCluster(int div1, int div2, int level, double distance) {
        clusters.get(div1).addMember(new Cluster(clusters.get(div2).getInstance()));
        clusters.get(div1).setLevel(level);
        clusters.get(div1).setDistance(distance);
        clusters.remove(div2);
        
        for (int i = 0; i < distanceMatrix.size(); i++) {
            if (type == SINGLE) {
                if ( distanceMatrix.get(div1).get(i) > distanceMatrix.get(div2).get(i)) {
                    distanceMatrix.get(div1).set(i, distanceMatrix.get(div2).get(i));
                }
            } else if (type == COMPLETE ) {
                if ( distanceMatrix.get(div1).get(i) < distanceMatrix.get(div2).get(i)) {
                    distanceMatrix.get(div1).set(i, distanceMatrix.get(div2).get(i));
                }
            }
        }
        
        for (int i = 0; i < clusters.size(); i++) {
            distanceMatrix.get(i).remove(div2);
        }
        System.out.println(clusters.size()+","+distanceMatrix.get(0).size());
    }
    
    @Override
    public void buildClusterer(Instances instances) throws Exception {
        distFunc.setInstances(instances);
        if (instances.numInstances() == 0) {
            return;
        }
        for(int i = 0; i < instances.numInstances(); i++) {
            clusters.add(new Cluster(instances.instance(i)));
        }
        for (int i = 0; i < clusters.size(); i++) {
            ArrayList distanceRow = new ArrayList<Double>();
            for (int j = 0; j < clusters.size(); j++) {
                distanceRow.add(distFunc.distance(clusters.get(i).getInstance(), clusters.get(j).getInstance()));
            }
            distanceMatrix.add(distanceRow);
        }
        
        int clusterCounter = 0;
        System.out.println(clusters.size()+","+distanceMatrix.get(0).size());
        while (clusters.size() > numClusters) {
            double val = Double.MAX_VALUE;
            int div1 = -1;
            int div2 = -1;
            for (int i = 0; i < distanceMatrix.size(); i++) {
                for (int j = i + 1; j < distanceMatrix.size(); j++) {
                    if (distanceMatrix.get(i).get(j) < val) {
                        val = distanceMatrix.get(i).get(j);
                        div1 = i;
                        div2 = j;
                    }   
                }
            }
            clusterCounter++;
            mergeCluster(div1, div2, clusterCounter, val);
        }
    }

    @Override
    public int clusterInstance(Instance instance) throws Exception {
        double min = Double.MAX_VALUE;
        int clusterNum = -1;
        for (int i = 0; i < clusters.size(); i++) {
            double tempMin = distFunc.distance(clusters.get(i).getInstance(), instance);
            if (tempMin < min) {
                min = tempMin;
                clusterNum = i;
            }
            ArrayList<Cluster> cluster = clusters.get(i).getMembers();
            for (int j = 0; j < cluster.size(); j++) {
                tempMin = distFunc.distance(clusters.get(j).getInstance(), instance);
                if (tempMin < min) {
                    min = tempMin;
                    clusterNum = i;
                }
            }
        }
        return clusterNum;
    }
    
    @Override
    public int numberOfClusters() throws Exception {
        return numClusters;
    }
    
    @Override
    public String toString() {
        if (clusters.isEmpty())
            return "myAgnes: No model built yet.";
        String summary = new String();
        for (int i = 0; i < clusters.size(); i++) {
            int count = 1 + clusters.get(i).getMembers().size();
            summary += "Cluster " + i + " has " + count + " members.\n";
        }
        return summary;
    }
}
