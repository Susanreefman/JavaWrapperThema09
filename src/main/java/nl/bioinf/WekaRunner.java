package nl.bioinf;

import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.trees.J48;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

import java.io.IOException;

/**
 * If you saved a model to a file in WEKA, you can use it reading the generated java object.
 * Here is an example with Random Forest classifier (previously saved to a file in WEKA):
 * import java.io.ObjectInputStream;
 * import weka.core.Instance;
 * import weka.core.Instances;
 * import weka.core.Attribute;
 * import weka.core.FastVector;
 * import weka.classifiers.trees.RandomForest;
 * RandomForest rf = (RandomForest) (new ObjectInputStream(PATH_TO_MODEL_FILE)).readObject();
 * <p>
 * or
 * RandomTree treeClassifier = (RandomTree) SerializationHelper.read(new FileInputStream("model.weka")));
 *
 * Instances is een weka class.
 * DataSource is een weka class.
 * getDataSet is een weka method
 */
public class WekaRunner {
    private final String modelFile = "testdata/naiveBayes.model";

    /**
     *
     *
     */
    public static void main(String[] args) {
        WekaRunner runner = new WekaRunner();
        runner.start();
    }

    /**
     *
     *
     */
    private void start() {
        String datafile = "testdata/data_teds.arff";
        String unknownFile = "testdata/unknown_data_teds.arff";
        try {
            // Building the model
            Instances instances = loadArff(datafile);
            printInstances(instances);
            // J48 is public class weka.classifiers.trees
            NaiveBayes naiveBayes = buildClassifier(instances);
            saveClassifier(naiveBayes);
            NaiveBayes fromFile = loadClassifier();

            // Using the model
            Instances unknownInstances = loadArff(unknownFile);
            System.out.println("\n unclassified unknownInstances = \n" + unknownInstances);
            classifyNewInstance(fromFile, unknownInstances);

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    /**
     * classifyNewInstance - using the model printing the instances classified by the model
     * @param tree - the model
     * @param unknownInstances - data with to be classified instances with the model
     */
    private void classifyNewInstance(NaiveBayes tree, Instances unknownInstances) throws Exception {
        // create copy
        Instances labeled = new Instances(unknownInstances);
        // label instances
        for (int i = 0; i < unknownInstances.numInstances(); i++) {
            double clsLabel = tree.classifyInstance(unknownInstances.instance(i));
            labeled.instance(i).setClassValue(clsLabel);
        }
        System.out.println("\n New, labeled = \n" + labeled);
    }


    /**
     * loadClassifier - reads saved classifier
     * @return Class with model from file
     */
    private NaiveBayes loadClassifier() throws Exception {
        // deserialize model
        return (NaiveBayes) weka.core.SerializationHelper.read(modelFile);
    }

    /**
     * saveClassifier - serialize tree to the modelFile
     * @param naiveBayes - built tree
     */
    private void saveClassifier(NaiveBayes naiveBayes) throws Exception {
        //post 3.5.5
        // serialize model
        weka.core.SerializationHelper.write(modelFile, naiveBayes);

        // serialize model pre 3.5.5
//        ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(modelFile));
//        oos.writeObject(j48);
//        oos.flush();
//        oos.close();
    }

    /**
     * buildClassifier
     * @param instances - data
     * @return tree - fully tree build by Classifier in Weka
     */
    private NaiveBayes buildClassifier(Instances instances) throws Exception {
        String[] options = new String[1];
        options[0] = "-D";            // unpruned tree
        NaiveBayes tree = new NaiveBayes();         // new instance of tree
        tree.setOptions(options);     // set the options
        tree.buildClassifier(instances);   // build classifier
        return tree;
    }

    /**
     * printInstances printing attributes, class index and instances
     * @param instances - data
     */
    private void printInstances(Instances instances) {
        int numAttributes = instances.numAttributes();

        for (int i = 0; i < numAttributes; i++) {
            System.out.println("attribute " + i + " = " + instances.attribute(i));
        }
        System.out.println("class index = " + instances.classIndex());

        int numInstances = instances.numInstances();
        for (int i = 0; i < numInstances; i++) {
            if (i == 5) break;
            Instance instance = instances.instance(i);
            System.out.println("instance = " + instance);
        }
    }

    /**
     * Instances loadArff - load file
     * @param datafile - String name of datafile from input
     * @return data
     */
    private Instances loadArff(String datafile) throws IOException {
        try {
            // loading data from files
            DataSource source = new DataSource(datafile);
            // returns full dataset, can be null in case of an error
            Instances data = source.getDataSet();

            // setting class attribute if the data format does not provide this information
           if (data.classIndex() == -1)
                data.setClassIndex(data.numAttributes() - 1);
            return data;
        } catch (Exception e) {
            throw new IOException("could not read from file");
        }
    }
}