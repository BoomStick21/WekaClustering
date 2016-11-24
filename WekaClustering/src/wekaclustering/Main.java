package wekaclustering;

import java.io.File;
import java.io.FileInputStream;
import java.util.ArrayList;
import java.util.Random;
import java.util.Scanner;
import java.util.logging.Level;
import java.util.logging.Logger;
import weka.clusterers.AbstractClusterer;
import weka.core.*;
import weka.core.converters.*;
import weka.clusterers.ClusterEvaluation;
import weka.clusterers.Clusterer;
import weka.clusterers.HierarchicalClusterer;
import weka.clusterers.SimpleKMeans;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

public class Main {
    private static Instances dataSet;
    private static AbstractClusterer cls;
    private static String classifierType;
    private static String[] optionsArr;
    
    public static void loadFile() throws Exception {
        ArrayList<String> fileNames = getFileNames("input/train");
        System.out.println("Available Training Set : ");
        for (int i = 0; i < fileNames.size(); i++) {
            System.out.println(i + ". " + fileNames.get(i));
        }
        System.out.print("Select the number of the training set to load : ");
        Scanner in = new Scanner(System.in);
        int choice = in.nextInt();
        if (choice > fileNames.size()-1) {
            System.out.println("Input error. Try again.");
            System.out.print("Select the number of the training set to load : ");
            choice = in.nextInt();
        }
        dataSet = ConverterUtils.DataSource.read("input/train/" + fileNames.get(choice));
        dataSet.setClassIndex(dataSet.numAttributes() - 1);
        Remove remove = new Remove();               
        remove.setAttributeIndices("" + (dataSet.classIndex() + 1));
        remove.setInputFormat(dataSet);
        dataSet = Filter.useFilter(dataSet, remove); 
//         if (dataSet.classIndex() == -1)
//            dataSet.setClassIndex(dataSet.numAttributes() - 1);
        System.out.println("Dataset "+ fileNames.get(choice) + " successfully loaded.\n");
    }
    
    public static ArrayList<String> getFileNames(String dir) {
        ArrayList<String> results = new ArrayList<>();
        File[] files = new File(dir).listFiles();
        

        for (File file : files) {
            if (file.isFile()) {
                results.add(file.getName());
            }
        }
        return results;
    }
    
    public static void command() throws Exception {
        if (classifierType != null) {
            System.out.println("Current Clusterer : " + classifierType);
        }
        System.out.println("Commands :");
        System.out.println("1. Load New DataSet");
        System.out.println("2. Print DataSet");
        System.out.println("3. Remove Attribute");
        System.out.println("4. Set Options");
        System.out.println("5. Create Clusterer");
        if (classifierType != null) {
            System.out.println("6. Print Result");
            System.out.println("7. Percentage Split");
            System.out.println("8. Use Training Set");
            System.out.println("9. Use Existing Test Set");
            System.out.println("10. Predict Test Set");
            System.out.println("11. Save Model");
            System.out.println("12. Load Model");
        }
        System.out.println("0. Exit");
        System.out.print("> ");
        Scanner in = new Scanner(System.in);
        int num = in.nextInt();
        if (classifierType == null) {
            if (num > 5) {
                System.out.println("Input error. Try again.");
                System.out.print("> ");
                num = in.nextInt();
            }
        } else {
            if (num > 12) {
                System.out.println("Input error. Try again.");
                System.out.print("> ");
                num = in.nextInt();
            }
        }
        switch(num) {
            case 1 : {
                loadFile();
                if (classifierType.equals("KMeans")) {
                    try {
                        cls = new SimpleKMeans();
                        cls.buildClusterer(dataSet);
                    } catch (Exception ex) {
                        System.out.println(ex);
                    }
                } else if (classifierType.equals("Hierarchical")) {
                    try {
                        cls = new HierarchicalClusterer();
                        cls.buildClusterer(dataSet);
                    } catch (Exception ex) {
                        System.out.println(ex);
                    }
                } else if (classifierType.equals("MyKMeans")) {
                    try {
                        cls = new myKMeans();
                        cls.buildClusterer(dataSet);
                    } catch (Exception ex) {
                        System.out.println(ex);
                    }
                } else if (classifierType.equals("MyAgnes")) {
                    try {
                        cls = new myAgnes();
                        if (optionsArr != null) {
                            ((myAgnes)cls).setOptions(optionsArr);
                        }
                        cls.buildClusterer(dataSet);
                    } catch (Exception ex) {
                        System.out.println(ex);
                    }
                }
                break;
            }
            case 2 : {
                System.out.println(dataSet.toString());
                break;
            }
            case 3 : {
                System.out.print("Enter which attribute to remove : ");
                int index = in.nextInt();
                Attribute attribute = dataSet.attribute(index);
                dataSet.deleteAttributeAt(index);
                System.out.println("Removed attribute : " + attribute.toString());
                break;
            }
            case 4 : {
                System.out.println("Options : ");
                ArrayList<String> options = new ArrayList<>();
                in = new Scanner(System.in);
                
                System.out.print("-N (default 2) : ");
                String value = in.nextLine();
                if (value.isEmpty()) value = "2";
                options.add("-N");
                options.add(value);
                
                System.out.print("-L (SINGLE/COMPLETE) : ");
                value = in.nextLine();
                options.add("-L");
                options.add(value);
                
                optionsArr = options.toArray(new String[options.size()]);
                
                break;            
            }
            case 5 : {
                System.out.println("Commands :");
                System.out.println("1. Simple KMeans");
                System.out.println("2. Hierarchical Clusterer");
                System.out.println("3. MyKMeans");
                System.out.println("4. MyAgnes");
                System.out.println("0. Return");
                System.out.print("> ");
                int choice = in.nextInt();
                switch(choice) {
                    case 1 : {
                        try {
                            cls = new SimpleKMeans();
                            cls.buildClusterer(dataSet);
                            classifierType = "KMeans";
                            System.out.println("SimpleKMeans Clusterer successfully built.\n");
                        } catch(Exception ex) {
                            System.out.println(ex);
                        }
                        break;
                    }
                    case 2 : {
                        try {
                            cls = new HierarchicalClusterer();
                            cls.buildClusterer(dataSet);
                            classifierType = "Hierarchical";
                            System.out.println("Hierarchical Clusterer successfully built.\n");
                        } catch(Exception ex) {
                            System.out.println(ex);
                        }
                        break;
                    }
                    case 3 : {
                        try {
                            cls = new myKMeans();
                            cls.buildClusterer(dataSet);
                            classifierType = "MyKMeans";
                            System.out.println("MyKMeans Clusterer successfully built.\n");
                        } catch(Exception ex) {
                            break;
                        }
                    }
                    case 4 : {
                        try {
                            cls = new myAgnes();
                            if (optionsArr != null) {
                                ((myAgnes)cls).setOptions(optionsArr);
                            }
                            cls.buildClusterer(dataSet);
                            classifierType = "MyAgnes";
                            System.out.println("MyAgnes Clusterer successfully built.\n");
                        } catch(Exception ex) {
                            System.out.println(ex);
                        }
                        break;
                    }                    
                    default : {
                        break;
                    }
                }
                break;
            }
            case 6 : {
                System.out.println(cls.toString());
                break;
            }
            case 7 : {
                System.out.print("Percentage of Training Set : ");
                float percentTrain = in.nextFloat();
                if ((percentTrain <= 0.0) || (percentTrain >= 100.0)) {
                    System.out.println("Input error. Try again.");
                    System.out.print("> ");
                    percentTrain = in.nextFloat();
                }
                float percentTest = 100 - percentTrain;                        
                System.out.println("Percentage of Test Set : " + percentTest);
                int trainSize = (int) Math.round(dataSet.numInstances() * percentTrain / 100);
                int testSize = dataSet.numInstances() - trainSize;
                dataSet.randomize(new java.util.Random(1));
                Instances train = new Instances(dataSet, 0, trainSize);
                Instances test = new Instances(dataSet, trainSize, testSize);
                if (classifierType.equals("MyAgnes"))
                    cls = new myAgnes();
                cls.buildClusterer(train);
                ClusterEvaluation eval = new ClusterEvaluation();
                eval.setClusterer(cls);
                eval.evaluateClusterer(test);
                System.out.println(eval.clusterResultsToString());
                break;
            }
            case 8 : {
                ClusterEvaluation eval = new ClusterEvaluation();
                if (classifierType.equals("MyAgnes")) {
                    cls = new myAgnes();
                    cls.buildClusterer(dataSet);
                }
                eval.setClusterer(cls);
                eval.evaluateClusterer(dataSet);      
                System.out.println(eval.clusterResultsToString());
                break;
            }
            case 9 : {
                ArrayList<String> fileNames = getFileNames("input/test");
                System.out.println("Available Tests : ");
                for (int i = 0; i < fileNames.size(); i++) {
                    System.out.println(i + ". " + fileNames.get(i));
                }
                System.out.print("Select the number of the test to load : ");
                int choice = in.nextInt();
                if (choice > fileNames.size()-1) {
                    System.out.println("Input error. Try again.");
                    System.out.print("Select the number of the test to load : ");
                    choice = in.nextInt();
                }
                Instances test = ConverterUtils.DataSource.read("input/test/" + fileNames.get(choice));
                if (test.classIndex() == -1)
                    test.setClassIndex(test.numAttributes() - 1);
                if (classifierType.equals("MyAgnes"))
                    cls = new myAgnes();
                cls.buildClusterer(dataSet);
                ClusterEvaluation eval = new ClusterEvaluation();
                eval.setClusterer(cls);
                eval.evaluateClusterer(test);      
                System.out.println(eval.clusterResultsToString());
                break;
            }
            case 10 : {
                ArrayList<String> fileNames = getFileNames("input/test");
                System.out.println("Available Tests : ");
                for (int i = 0; i < fileNames.size(); i++) {
                    System.out.println(i + ". " + fileNames.get(i));
                }
                System.out.print("Select the number of the test to load : ");
                int choice = in.nextInt();
                if (choice > fileNames.size()-1) {
                    System.out.println("Input error. Try again.");
                    System.out.print("Select the number of the test to load : ");
                    choice = in.nextInt();
                }
                Instances test = ConverterUtils.DataSource.read("input/test/" + fileNames.get(choice));
                if (test.classIndex() == -1)
                    test.setClassIndex(test.numAttributes() - 1);
                for (int i = 0; i < test.numInstances(); i++) {
                    double label = cls.clusterInstance(test.instance(i));
                    test.instance(i).setClassValue(label);
                    for (int j = 0; j < test.instance(i).numValues()-1; j++) {
                        if (test.instance(i).attribute(j).isNominal()) {
                            System.out.print(test.instance(i).stringValue(j)+",");
                        }
                        else {
                            System.out.print(test.instance(i).value(j)+",");
                        }
                    }
                    System.out.println(test.instance(i).stringValue(test.instance(i).numValues()-1));
		}
                break;
            }
            case 11 : {
                System.out.print("Name of your model : ");
                Scanner nameScan = new Scanner(System.in);
                String name = nameScan.nextLine();
                File dir = new File("saves/" + classifierType + "/");
                dir.mkdirs();
                SerializationHelper.write("saves/" + classifierType + "/" + name + ".model", cls);
                System.out.println("File " + name + ".model successfully created.\n");
                break;
            }
            case 12 : {
                ArrayList<String> fileNames = getFileNames("saves/" + classifierType);
                System.out.println("Saved Models : ");
                for (int i = 0; i < fileNames.size(); i++) {
                    System.out.println(i + ". " + fileNames.get(i));
                }
                System.out.print("Select the number of the model to load : ");
                int choice = in.nextInt();
                if (choice > fileNames.size()-1) {
                    System.out.println("Input error. Try again.");
                    System.out.print("Select the number of the model to load : ");
                    choice = in.nextInt();
                }
                if (classifierType.equals("KMeans"))
                    cls = (SimpleKMeans) SerializationHelper.read(
                            new FileInputStream("saves/KMeans/" + fileNames.get(choice)));
                else if (classifierType.equals("Hierarchical"))
                    cls = (HierarchicalClusterer) SerializationHelper.read(
                            new FileInputStream("saves/Hierarchical/" + fileNames.get(choice)));
                else if (classifierType.equals("MyKMeans"))
                    cls = (HierarchicalClusterer) SerializationHelper.read(
                            new FileInputStream("saves/MyKMeans/" + fileNames.get(choice))); 
                else if (classifierType.equals("MyAgnes"))
                    cls = (HierarchicalClusterer) SerializationHelper.read(
                            new FileInputStream("saves/MyAgnes/" + fileNames.get(choice)));                 
                System.out.println(fileNames.get(choice) + " successfully loaded.\n");
                break;
            }
            case 0 : {
                System.exit(1);
                break;
            }
        }
    }

    public static void main(String[] args){
        try {
            loadFile();
            while (true) {
              command();  
            }
        } catch (Exception ex) {
            Logger.getLogger(Main.class.getName()).log(Level.SEVERE, null, ex);
        }
     
    }
}

