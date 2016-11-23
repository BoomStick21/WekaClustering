/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package wekaclustering;

import java.util.ArrayList;
import weka.core.Instance;


public class Cluster {
    ArrayList<Cluster> members;
    Instance instance;
    double distance;
    int level;
    
    public Cluster(Instance i) {
        instance = i;
        level = -1;
        distance = Double.MAX_VALUE;
        members = new ArrayList<Cluster>();
    }
    
    public Instance getInstance() {
        return instance;
    }
    
    public ArrayList<Cluster> getMembers() {
        return members;
    }
    
    public void addMember(Cluster c) {
        members.add(c);
    }
    
    public void setLevel(int _level) {
        level = _level;
    }
    
    public void setDistance(double _dist) {
        distance = _dist;
    }
}
