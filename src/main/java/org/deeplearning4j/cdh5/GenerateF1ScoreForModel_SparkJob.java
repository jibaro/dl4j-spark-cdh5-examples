package org.deeplearning4j.cdh5;

import java.io.BufferedOutputStream;
import java.io.BufferedReader;
import java.io.DataInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.util.Arrays;

import org.apache.commons.io.FileUtils;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics;
import org.apache.spark.mllib.evaluation.MulticlassMetrics;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.util.MLUtils;
import org.apache.spark.rdd.RDD;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.RBM;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import scala.Tuple2;
import scala.reflect.ClassManifestFactory;

public class GenerateF1ScoreForModel_SparkJob {


	public static String loadTextFileFromHDFS(org.apache.hadoop.conf.Configuration hadoopConfig, FileSystem hdfs, String path) throws IllegalArgumentException, IOException {
		

        StringBuilder textBuffer = new StringBuilder();
        FSDataInputStream hdfsInputStream = hdfs.open(new Path( path ));
        BufferedReader br = new BufferedReader(new InputStreamReader( hdfsInputStream ));
        
        String line = br.readLine();
        while (line != null) {
        	
        	textBuffer.append(line);
        	line = br.readLine();
            
        }
        
        br.close();
        hdfsInputStream.close();		
		
        return textBuffer.toString();
		
	}
	
	
    public static void main( String[] args) throws Exception {
    	
        Nd4j.MAX_SLICES_TO_PRINT = -1;
        Nd4j.MAX_ELEMENTS_PER_SLICE = -1;
        Nd4j.ENFORCE_NUMERICAL_STABILITY = true;    	
    	
        //List<String> labels = Arrays.asList("beach", "desert", "forest", "mountain", "rain", "snow");

        final int numRows = 1;
        final int numColumns = 17;
        int outputNum = 2; //labels.size();
        //int batchSize = 150;
        //int iterations = 5;
        //int splitTrainNum = (int) (batchSize * .8);
        //int seed = 123;
        
        if (args.length < 5) {
        	System.out.println("Need at least an hdfs path");
        	return;
        }
        
        org.apache.hadoop.conf.Configuration hadoopConfig = new org.apache.hadoop.conf.Configuration();
        FileSystem hdfs = FileSystem.get(hadoopConfig);
        
        
        String hdfsPathString = args[0];

        // .setMaster("local[*]")
        JavaSparkContext sc = new JavaSparkContext(new SparkConf().setAppName("GenerateF1ScoreForModel"));
        
        String hdfsFilesToEvaluateModel = hdfsPathString; //"hdfs:///user/cloudera/svmlight/*";
        String hdfsPathToModelJSON = args[1];
        String hdfsPathToModelParams = args[2];
        String hdfsPathToEvalOutput = args[3];
        String svmLightFeatureColumns = args[4];
        int parsedSVMLightFeatureColumns = Integer.parseInt( svmLightFeatureColumns );
        if (0 == parsedSVMLightFeatureColumns) {
        	System.err.println("Bad SVMLight feature count: " + svmLightFeatureColumns);
        }
        

        String jsonModelConf = loadTextFileFromHDFS( hadoopConfig, hdfs, hdfsPathToModelJSON ); 
        
        
        System.out.println( "\n--------------------\n" );
        System.out.println( jsonModelConf );
        System.out.println( "\n--------------------\n" );
        
        MultiLayerConfiguration confFromJson = MultiLayerConfiguration.fromJson( jsonModelConf );        
        
        FSDataInputStream hdfsInputStream_ModelParams = hdfs.open(new Path( hdfsPathToModelParams ));
        //BufferedReader br = new BufferedReader(new InputStreamReader( hdfsInputStream ));
        
        
        DataInputStream dis = new DataInputStream( hdfsInputStream_ModelParams );
        INDArray newParams = Nd4j.read( dis );
        dis.close();
        MultiLayerNetwork savedNetwork = new MultiLayerNetwork( confFromJson );
        savedNetwork.init();
        savedNetwork.setParameters(newParams);
        //System.out.println("Original network params " + model.params());
        System.out.println(savedNetwork.params());

        
        //Evaluation eval = new Evaluation(outputNum);
        
        
        RDD<LabeledPoint> evaluate_svmLight_data_rdd = MLUtils.loadLibSVMFile( sc.sc(), hdfsFilesToEvaluateModel, true, parsedSVMLightFeatureColumns );
        
        System.out.println( "\n\n\n" +  evaluate_svmLight_data_rdd.first() + "\n" );
        
        JavaRDD<LabeledPoint> evaluate_svmLight_data_JavaRDD = JavaRDD.fromRDD( evaluate_svmLight_data_rdd, ClassManifestFactory.fromClass(LabeledPoint.class) );
        
        
        /*
        StandardScaler scaler = new StandardScaler(true,true);

        final StandardScalerModel scalarModel = scaler.fit(data.map(new Function<LabeledPoint, Vector>() {
            @Override
            public Vector call(LabeledPoint v1) throws Exception {
                return v1.features();
            }
        }).rdd());
*/
        
        /*
         * Can we use local variables here?
         * 
         * TODO: how do we sum the output of all the guesses into one place?
         * 
         */
/*        //get the trained data for the train/test split
        JavaRDD<LabeledPoint> demoOutput = evaluate_svmLight_data_JavaRDD.map(new Function<LabeledPoint, LabeledPoint>() {
            @Override
            public LabeledPoint call(LabeledPoint v1) throws Exception {
                Vector features = v1.features();
                //Vector normalized = scalarModel.transform(features);
                return new LabeledPoint( v1.label(), v1.features() );
            }
        }).cache();
*/
  //      System.out.println( "Scanned and Cached the dataset " );
        
        /*
        System.out.println( "\n\n> ---------------------- [ Calculating F1 Score ] ------------------- \n\n\n" ); 
        
     // Compute raw scores on the test set.
        JavaRDD<Tuple2<Object, Object>> predictionAndLabels = data.map(
                new Function<LabeledPoint, Tuple2<Object, Object>>() {
                    public Tuple2<Object, Object> call(LabeledPoint p) {
                        Vector prediction = trainedNetworkWrapper.predict(p.features());
                        double max = 0;
                        double idx = 0;
                        for(int i = 0; i < prediction.size(); i++) {
                            if(prediction.apply(i) > max) {
                                idx = i;
                                max = prediction.apply(i);
                            }
                        }

                        return new Tuple2<Object, Object>(idx, p.label());
                    }
                }
        );        
        */
     
        
        final SparkDl4jMultiLayer trainedNetworkWrapper = new SparkDl4jMultiLayer(sc.sc(),savedNetwork);
/*
        // Compute raw scores on the test set.
        JavaRDD<Tuple2<Object, Object>> predictionAndLabels = evaluate_svmLight_data_JavaRDD.map(
                new Function<LabeledPoint, Tuple2<Object, Object>>() {
                    public Tuple2<Object, Object> call(LabeledPoint p) {
                        Vector prediction = trainedNetworkWrapper.predict(p.features());
                        
                        return new Tuple2<Object, Object>(prediction, p.label());
                    }
                }
        );
*/
        JavaRDD<Tuple2<Object, Object>> predictionAndLabels = evaluate_svmLight_data_JavaRDD.map(
                new Function<LabeledPoint, Tuple2<Object, Object>>() {
                    public Tuple2<Object, Object> call(LabeledPoint p) {
                        Vector prediction = trainedNetworkWrapper.predict(p.features());
                        double max = 0;
                        double idx = 0;
                        for(int i = 0; i < prediction.size(); i++) {
                            if(prediction.apply(i) > max) {
                                idx = i;
                                max = prediction.apply(i);
                            }
                        }

                        return new Tuple2<Object, Object>(idx, p.label());
                    }
                }
        );        



        // Get evaluation metrics.
        MulticlassMetrics metrics = new MulticlassMetrics( predictionAndLabels.rdd() );
        double precision = metrics.fMeasure();
        //BufferedOutputStream bos = new BufferedOutputStream(new FileOutputStream(""));
        //Nd4j.write(bos,trainedNetwork.params());
        //FileUtils.write(new File("conf.yaml"),trainedNetwork.conf().toYaml());
        //System.out.println("F1 = " + precision);        
        
        JavaRDD<String> metricsResult = sc.parallelize(Arrays.asList(
        	       "Precision: " + metrics.precision(), 
        	       "\nRecall: " +metrics.recall(),
        	       "\nF-Measure: " +metrics.fMeasure(), 
        	       "\nConfusion metrics: \n" + metrics.confusionMatrix()));    
        
        System.out.println( "Precision: " + metrics.precision() );
        System.out.println( "Recall: " + metrics.recall() );
        System.out.println( "F1: \n" + metrics.fMeasure() );
        System.out.println( "\nConfusion metrics: \n" + metrics.confusionMatrix() );
        
        metricsResult.saveAsTextFile( hdfsPathToEvalOutput );
        
        
        
        System.out.println( "Predictions complete..." );    
        
        // save F1 informationt to HDFS

/*
        
        String modelPathLocation =  "hdfs:///user/cloudera/dl4j_bank_model.bin";
        
        org.apache.hadoop.conf.Configuration hadoopConfig = new org.apache.hadoop.conf.Configuration();
        FileSystem hdfs = FileSystem.get(hadoopConfig);
        Path modelPath = new Path( modelPathLocation );
        
        if ( hdfs.exists( modelPath )) {
        	hdfs.delete( modelPath, true ); 
        } 
        
        OutputStream os = hdfs.create( modelPath );
        
        
        BufferedOutputStream bos = new BufferedOutputStream( os );
        Nd4j.write(bos,trainedNetwork.params());
      //  FileUtils.write(new File("conf.yaml"),trainedNetwork.conf().toYaml());
      //  System.out.println("F1 = " + precision);
        bos.close();
        hdfs.close();
        
        System.out.println( "Saving model to: " + modelPath );
        
        System.out.println( "\n\n> ---------------------- Saving Model [ done ] ------------------- \n\n\n" );
*/

    }		
	
}
