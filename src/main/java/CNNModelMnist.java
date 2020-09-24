import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.deeplearning4j.util.ModelSerializer;

import java.io.File;
import java.io.IOException;
import java.util.Random;
import java.util.concurrent.Callable;

public class CNNModelMnist {

    private static Logger logger= LoggerFactory.getLogger(CNNModelMnist.class);

    public static void main(String[] args) throws IOException, InterruptedException {

        long seed =1234;
        double learningRate=0.001;
        long height=28;
        long width=28;
        long depth=1; // la profondeur => 1 parce que image blanc et noir
        int outputSize= 10;
        Random randomGenNum=new Random(seed);
        //double quadraticError=0.0005;



        // LA CONFIGURATION DU MODEL

        MultiLayerConfiguration configuration = new NeuralNetConfiguration.Builder()
                .seed(seed) // concenre generation des nombre aleatoire
                .updater(new Adam(learningRate))  // vitesse d'apprentissage
                .list()  // la liste des couche => layers
                .setInputType(InputType.convolutionalFlat(height,width,depth)) // le type des entrées
                .layer(0, new ConvolutionLayer.Builder() ///// convolution////////////////////
                        .nIn(depth) // nmbr image
                        .nOut(20) // nombre de filtre (kernal)
                        .activation(Activation.RELU)  // fct activation (fct RELU) => ecrasser les nombre negative
                        .kernelSize(5,5) // la taille du filtre => kernal
                        .stride(1,1) // glisser d'un 1 px vertical et 1 px horz
                        .build())
                .layer(1, new SubsamplingLayer.Builder() ////////// max pulling/////////
                        .kernelSize(2,2)
                        .stride(2,2)
                        .poolingType(SubsamplingLayer.PoolingType.MAX)
                        .build())
                .layer(2, new ConvolutionLayer.Builder() ///// convolution////////////////////
                        .nOut(50)
                        .activation(Activation.RELU)
                        .kernelSize(5,5)
                        .stride(1,1)
                        .build())
                .layer(3, new SubsamplingLayer.Builder() ////////// max pulling/////////
                        .kernelSize(2,2)
                        .stride(2,2)
                        .poolingType(SubsamplingLayer.PoolingType.MAX)
                        .build())
                .layer(4, new DenseLayer.Builder() ////// couche fully connected (ligne verticle) ///////////
                        .nOut(500)
                        .activation(Activation.RELU)
                        .build()) 
                .layer(5, new OutputLayer.Builder() ////////// output layer/////////
                        .nOut(outputSize)
                        .activation(Activation.SOFTMAX) /// LA PROB C'est 1 ou 0
                        .lossFunction(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)  // pour minimiser l'erreur
                        .build())
                .build();

        // afficher la config format json
        // System.out.println(configuration.toJson());

        // LE MODEL

        MultiLayerNetwork model = new MultiLayerNetwork(configuration);
        model.init();

        // entrainement du model
        System.out.println("Model training ....");

        //////////////////// lire les données /////////////////
        String path = System.getProperty("user.home")+"/mnist_png";
        System.out.println("hahwa path:"+path);
        File fileTrain =new File(path+"/testing");
        FileSplit fileSplitTrain = new FileSplit(fileTrain, NativeImageLoader.ALLOWED_FORMATS, randomGenNum);
        RecordReader recordReaderTrain = new ImageRecordReader(height,width,depth, new ParentPathLabelGenerator());

        recordReaderTrain.initialize(fileSplitTrain);
        // le nombre d'image en entrée => batch
        // index 0  c'est image et index 1 c'est label
        int batchSize=54;
        DataSetIterator dataSetIteratorTrain = new RecordReaderDataSetIterator(recordReaderTrain,batchSize,1,outputSize);
        //////////////////// end lire les données /////////////////

        DataNormalization scaler = new ImagePreProcessingScaler(0,1);
        dataSetIteratorTrain.setPreProcessor(scaler);
        logger.info("Neural Network Model Configuation");

        //////////////////// entrainer le model /////////////////
        int numEpoch=1;

            // LA VISUALISATION ui du model////////////////////
            UIServer uiServer = UIServer.getInstance();
            StatsStorage statsStorage = new InMemoryStatsStorage();
            uiServer.attach(statsStorage);
            model.setListeners(new StatsListener(statsStorage));
                    // le model doit cinverger vers 0
            // end LA VISUALISATION ui du model////////////////////

        for (int i = 0; i <numEpoch ; i++) {
            model.fit(dataSetIteratorTrain); // entrainer le model
        }

        System.out.println("Evaluation du Model ....");

        File fileTest =new File(path+"/testing");
        FileSplit fileSplitTrainTest = new FileSplit(fileTest, NativeImageLoader.ALLOWED_FORMATS, randomGenNum);
        RecordReader recordReaderTest = new ImageRecordReader(height,width,depth, new ParentPathLabelGenerator());
        recordReaderTest.initialize(fileSplitTrainTest);
        DataSetIterator dataSetIteratorTrainTest = new RecordReaderDataSetIterator(recordReaderTest,batchSize,1,outputSize);
        DataNormalization scalerTest = new ImagePreProcessingScaler(0,1);
        dataSetIteratorTrainTest.setPreProcessor(scalerTest);

        Evaluation evaluation= new Evaluation();
        // boucle pour les batchs
        while (dataSetIteratorTrainTest.hasNext()){
            DataSet dataSet = dataSetIteratorTrainTest.next();
            INDArray features = dataSet.getFeatures();
            INDArray targetLabels = dataSet.getLabels();

            INDArray pretected = model.output(features);

            evaluation.eval(pretected, targetLabels);
        }
        System.out.println(evaluation.stats());


        // !! dataset contient bach et le bach contient des images

        /*
        while (dataSetIteratorTrain.hasNext()){
            DataSet dataSet = dataSetIteratorTrain.next();
            INDArray features = dataSet.getFeatures();
            INDArray lables = dataSet.getLabels();
            System.out.println(features.shapeInfoToString());
            System.out.println(lables.shapeInfoToString());
            System.out.println(lables);
            System.out.println("************************");
        }
        */


        logger.info("Saving model ....");
        ModelSerializer.writeModel(model,new File(path+"/model.zip"),true);

    }
}
