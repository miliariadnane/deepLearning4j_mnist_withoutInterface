import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.io.File;

public class sc {
    /*
      System.out.println("Evaluation du Model ....");

    File fileTrainTest =new File(basePath+"/testing");
    FileSplit fileSplitTrainTest = new FileSplit(fileTrainTest, NativeImageLoader.ALLOWED_FORMATS, randomGenNum);
    RecordReader recordReaderTrainTest = new ImageRecordReader(height,width,depth, new ParentPathLabelGenerator());
        recordReaderTrain.initialize(fileSplitTrainTest);
    DataSetIterator dataSetIteratorTrainTest = new RecordReaderDataSetIterator(recordReaderTrainTest,batchSize,1,outputSize);
        dataSetIteratorTrainTest.setPreProcessor(scaler);

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

     */
}
