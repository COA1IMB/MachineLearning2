import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.collection.ListStringRecordReader;
import org.datavec.api.split.ListStringSplit;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.earlystopping.EarlyStoppingResult;
import org.deeplearning4j.earlystopping.saver.LocalFileModelSaver;
import org.deeplearning4j.earlystopping.scorecalc.DataSetLossCalculator;
import org.deeplearning4j.earlystopping.termination.MaxEpochsTerminationCondition;
import org.deeplearning4j.earlystopping.termination.MaxTimeIterationTerminationCondition;
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingTrainer;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.TimeUnit;

public class MachineLearning2 {

    public static final int NUMBER_OF_DATA_SETS = 9999999;
    private static final int NUMBER_OF_COLUMNS = 30;
    private static final double LEARNING_RATE = 0.0001;
    private static final double MOMENTUM = 0.5;
    private static final int NUM_HIDDEN_NODES = 100;
    private static final int TRAINING_TIME = 100;
    private static final int MAX_EPOCHS = 200;
    private static final int MAX_EVAL_DEFAULTS = 100;
    private static final int MAX_EVAL_NON_DEFAULTS = 100;
    private static final String LEARN_FILE_PATH = "src\\main\\resources\\file_all_nn_prepared_07_630.csv";
    private static final String EVAL_FILE_PATH = "src\\main\\resources\\file_evaluation_nn_07_Q2_630.csv";

    public static void main(String[] args) {
        List<List<String>> data = getDataAsList();
        data = normalize(data);
        networkLearn(data);

        List<List<String>> dataEval = getEvalDataAsList();
        dataEval = normalizeEvalData(dataEval);
        evaluateNetwork(dataEval);
    }

    private static void networkLearn(List<List<String>> data) {
        int seed = 123;
        int batchSize = 50;
        int numInputs = 29;
        int numOutputs = 2;
        RecordReader rr = new ListStringRecordReader();

        try {
            rr.initialize(new ListStringSplit(data));
            //rr.initialize(new FileSplit(new File("src\\main\\resources\\file_all_nn_prepared07.csv")));
        } catch (Exception e) {
            System.out.println(e);
        }
        DataSetIterator trainIter = new RecordReaderDataSetIterator(rr, batchSize, 0, 2);

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
                .iterations(1)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .learningRate(LEARNING_RATE)
                .updater(org.deeplearning4j.nn.conf.Updater.NESTEROVS).momentum(MOMENTUM)
                .list()
                .layer(0, new DenseLayer.Builder()
                        .nIn(numInputs)
                        .nOut(NUM_HIDDEN_NODES)
                        .weightInit(WeightInit.XAVIER)
                        .activation("relu")
                        .build())
                .layer(1, new DenseLayer.Builder()
                        .nIn(NUM_HIDDEN_NODES)
                        .nOut(NUM_HIDDEN_NODES)
                        .weightInit(WeightInit.XAVIER)
                        .activation("relu")
                        .build())
                .layer(2, new DenseLayer.Builder()
                        .nIn(NUM_HIDDEN_NODES)
                        .nOut(NUM_HIDDEN_NODES)
                        .weightInit(WeightInit.XAVIER)
                        .activation("relu")
                        .build())
                .layer(3, new DenseLayer.Builder()
                        .nIn(NUM_HIDDEN_NODES)
                        .nOut(NUM_HIDDEN_NODES)
                        .weightInit(WeightInit.XAVIER)
                        .activation("relu")
                        .build())
                .layer(4, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .weightInit(WeightInit.XAVIER)
                        .activation("softmax")
                        .weightInit(WeightInit.XAVIER)
                        .nIn(NUM_HIDDEN_NODES)
                        .nOut(numOutputs)
                        .build()
                )
                .pretrain(false).backprop(true).build();

        // Apply Network and attach Listener to Web-UI
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(10));
        //UIServer uiServer = UIServer.getInstance();
        StatsStorage statsStorage = new InMemoryStatsStorage();
        model.setListeners(new StatsListener(statsStorage));
        //uiServer.attach(statsStorage);

        EarlyStoppingConfiguration esConf = new EarlyStoppingConfiguration.Builder()
                .epochTerminationConditions(new MaxEpochsTerminationCondition(MAX_EPOCHS))
                .iterationTerminationConditions(new MaxTimeIterationTerminationCondition(TRAINING_TIME, TimeUnit.MINUTES))
                .scoreCalculator(new DataSetLossCalculator(trainIter, true))
                .evaluateEveryNEpochs(1)
                .modelSaver(new LocalFileModelSaver("C:\\Users\\fabcot01\\IdeaProjects\\MachineLearning2\\"))
                .build();

        EarlyStoppingTrainer trainer = new EarlyStoppingTrainer(esConf, model, trainIter);
        EarlyStoppingResult result = trainer.fit();

        //Print out the results:
        System.out.println("Termination reason: " + result.getTerminationReason());
        System.out.println("Termination details: " + result.getTerminationDetails());
        System.out.println("Total epochs: " + result.getTotalEpochs());
        System.out.println("Best epoch number: " + result.getBestModelEpoch());
        System.out.println("Score at best epoch: " + result.getBestModelScore());

        File locationToSave = new File("C:\\Users\\fabcot01\\IdeaProjects\\MachineLearning2\\NeuralNetwork.zip");
        boolean saveUpdater = true;

        org.deeplearning4j.nn.api.Model model2 = result.getBestModel();

        try {
            ModelSerializer.writeModel(model2, locationToSave, saveUpdater);
        } catch (Exception e) {
            System.out.println(e.toString());
        }
    }

    private static List<List<String>> getDataAsList() {

        ArrayList<List<String>> data = null;
        try {
            String fileName = LEARN_FILE_PATH;
            BufferedReader br = null;
            String sCurrentLine;
            br = new BufferedReader(new FileReader(fileName));//file name with path
            data = new ArrayList<List<String>>();

            while ((sCurrentLine = br.readLine()) != null) {
                String[] parts1 = sCurrentLine.split(",");
                List<String> data2 = Arrays.asList(parts1);
                data.add(data2);
            }
            return data;
        } catch (IOException e) {
            e.printStackTrace();
        }
        return data;
    }

    private static List<List<String>> normalize(List<List<String>> data) {

        double[] values = new double[NUMBER_OF_DATA_SETS];
        double[] mins = new double[NUMBER_OF_COLUMNS];
        double[] maxs = new double[NUMBER_OF_COLUMNS];

        for (int j = 0; j < data.get(0).size(); j++) {
            for (int i = 0; i < data.size(); i++) {
                values[i] = Double.parseDouble(data.get(i).get(j));
            }
            mins[j] = Arrays.stream(values).min().getAsDouble();
            maxs[j] = Arrays.stream(values).max().getAsDouble();
        }
        for (List<String> temp : data) {
            for (int y = 0; y < 30; y++) {
                if (mins[y] != maxs[y]) {
                    double tempValue = (Double.parseDouble(temp.get(y)) - mins[y]) / (maxs[y] - mins[y]);

                    if (tempValue == 0.0) {
                        temp.set(y, "0");
                    } else if (tempValue == 1.0) {
                        temp.set(y, "1");
                    } else {
                        temp.set(y, Double.toString(tempValue));
                    }
                }
            }
        }
        return data;
    }

    private static List<List<String>> getEvalDataAsList() {

        ArrayList<List<String>> data = null;
        try {
            String fileName = EVAL_FILE_PATH;
            BufferedReader br = null;
            String sCurrentLine;
            br = new BufferedReader(new FileReader(fileName));//file name with path
            data = new ArrayList<List<String>>();

            while ((sCurrentLine = br.readLine()) != null) {
                String[] parts1 = sCurrentLine.split(",");
                List<String> data2 = Arrays.asList(parts1);
                data.add(data2);
            }
            return data;
        } catch (IOException e) {
            e.printStackTrace();
        }
        return data;
    }

    private static List<List<String>> normalizeEvalData(List<List<String>> data) {
        double[] values = new double[NUMBER_OF_DATA_SETS];
        double[] mins = new double[NUMBER_OF_COLUMNS];
        double[] maxs = new double[NUMBER_OF_COLUMNS];
        int counterDefaults = 0;
        int counterNonDefaults = 0;
        List<List<String>> data2 = new ArrayList<List<String>>();

        for (int j = 0; j < data.get(0).size(); j++) {
            for (int i = 0; i < data.size(); i++) {
                values[i] = Double.parseDouble(data.get(i).get(j));
            }
            mins[j] = Arrays.stream(values).min().getAsDouble();
            maxs[j] = Arrays.stream(values).max().getAsDouble();
        }
        for (List<String> temp : data) {
            for (int y = 0; y < 30; y++) {
                if (mins[y] != maxs[y]) {
                    double tempValue = (Double.parseDouble(temp.get(y)) - mins[y]) / (maxs[y] - mins[y]);

                    if (tempValue == 0.0) {
                        temp.set(y, "0");
                    } else if (tempValue == 1.0) {
                        temp.set(y, "1");
                    } else {
                        temp.set(y, Double.toString(tempValue));
                    }
                }
            }
            if(temp.get(0).equals("0") && MAX_EVAL_NON_DEFAULTS >= counterDefaults){
                counterDefaults++;
                System.out.println("Add Non Default");
                data2.add(temp);
            }else if(temp.get(0).equals("1") && MAX_EVAL_DEFAULTS >= counterNonDefaults){
                counterNonDefaults++;
                System.out.println("Add Default");
                data2.add(temp);
            }else{
                continue;
            }
        }
        return data2;
    }

    private static void evaluateNetwork(List<List<String>> dataEval) {
        RecordReader rrTest = new ListStringRecordReader();
        MultiLayerNetwork model = null;
        int batchSize = 10;
        int numOutputs = 2;

        try {
            rrTest.initialize(new ListStringSplit(dataEval));
        } catch (Exception e) {
            System.out.println(e);
        }

        DataSetIterator testIter = new RecordReaderDataSetIterator(rrTest, batchSize, 0, 2);

        try {
            model = ModelSerializer.restoreMultiLayerNetwork("C:\\Users\\fabcot01\\IdeaProjects\\MachineLearning2\\NeuralNetwork.zip");
        } catch (Exception e) {
            System.out.println(e.toString());
        }

        System.out.println("Evaluate model.......");
        Evaluation eval = new Evaluation(numOutputs);

        while (testIter.hasNext()) {
            DataSet t = testIter.next();
            INDArray features = t.getFeatureMatrix();
            INDArray lables = t.getLabels();
            INDArray predicted = model.output(features, false);
            eval.eval(lables, predicted);
        }
        System.out.println(eval.stats());
    }
}
