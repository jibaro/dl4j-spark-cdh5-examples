package org.deeplearning4j.cdh5;

import java.io.BufferedOutputStream;
import java.io.BufferedReader;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;

import org.apache.commons.io.FileUtils;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.util.MLUtils;
import org.apache.spark.rdd.RDD;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import scala.reflect.ClassManifestFactory;

public class BuildDL4JModel_SVMLight_SparkJob {


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
        
        if (args.length < 4) {
        	System.out.println("Need at least an hdfs path");
        	return;
        }
        
        org.apache.hadoop.conf.Configuration hadoopConfig = new org.apache.hadoop.conf.Configuration();
        FileSystem hdfs = FileSystem.get(hadoopConfig);

        
        String hdfsPathString = args[0];

        // .setMaster("local[*]")
        JavaSparkContext sc = new JavaSparkContext(new SparkConf().setAppName("BuildDL4JModel_SVMLight"));
        
        // training data location
        String hdfsFilesToTrainModel = hdfsPathString; //"hdfs:///user/cloudera/svmlight/*";
        
        // how we define network
        String hdfsPathToModelJSON = args[1];
        
        // where we want to save the model parameters
        String hdfsPathToModelParams = args[2];
        
        // how many features are in input data
        String svmLightFeatureColumns = args[3];
        int parsedSVMLightFeatureColumns = Integer.parseInt( svmLightFeatureColumns );
        if (0 == parsedSVMLightFeatureColumns) {
        	System.err.println("Bad SVMLight feature count: " + svmLightFeatureColumns);
        }
        
        
        
        /*
        Configuration svm_conf = new Configuration();
        svm_conf.set(SVMLightRecordReader.NUM_ATTRIBUTES, "17");
        
        SVMLightRecordReader recordReader = new SVMLightRecordReader();
        recordReader.setConf(svm_conf);
        
        
        JavaRDD<LabeledPoint> data = MLLibUtil.fromBinary(sc.binaryFiles( hdfsFiles ), recordReader);
        
        System.out.println( data.first() );
        */
        
        StringBuilder jsonBuffer = new StringBuilder();
        
        FSDataInputStream hdfsInputStream = hdfs.open(new Path(hdfsPathToModelJSON));
        BufferedReader br = new BufferedReader(new InputStreamReader( hdfsInputStream ));
        //hdfsInputStream.rea
        
        String line = br.readLine();
        //jsonBuffer.append(line);
        while (line != null) {
        	
        	jsonBuffer.append(line);
        	line = br.readLine();
            
            
        }
        
        br.close();
        hdfsInputStream.close();
        
        System.out.println( "\n--------------------\n" );
        System.out.println( jsonBuffer );
        System.out.println( "\n--------------------\n" );
        
        //MultiLayerConfiguration confFromJson = MultiLayerConfiguration.fromJson(FileUtils.readFileToString(new File( hdfsPathToModelJSON )));
        MultiLayerConfiguration confFromJson = MultiLayerConfiguration.fromJson( jsonBuffer.toString() );
        //DataInputStream dis = new DataInputStream(new FileInputStream( hdfsPathToModelParams ));
        //INDArray newParams = Nd4j.read( dis );
        //dis.close();
        MultiLayerNetwork savedNetwork = new MultiLayerNetwork( confFromJson );
        savedNetwork.init();
        //savedNetwork.setParameters(newParams);
        //System.out.println("Original network params " + model.params());
        //System.out.println(savedNetwork.params());

        
        
        
        
        RDD<LabeledPoint> evaluate_svmLight_data_rdd = MLUtils.loadLibSVMFile( sc.sc(), hdfsFilesToTrainModel, true, parsedSVMLightFeatureColumns );
        
        System.out.println( "\n\n\n" +  evaluate_svmLight_data_rdd.first() + "\n" );
        
        JavaRDD<LabeledPoint> evaluate_svmLight_data_JavaRDD = JavaRDD.fromRDD( evaluate_svmLight_data_rdd, ClassManifestFactory.fromClass(LabeledPoint.class) );
        
        
        SparkDl4jMultiLayer trainLayerSpark = new SparkDl4jMultiLayer( sc.sc(), confFromJson );
        //fit on the training set
        MultiLayerNetwork trainedNetwork = trainLayerSpark.fit( sc, evaluate_svmLight_data_JavaRDD );
        
        final SparkDl4jMultiLayer completedModelFromSparkToSave = new SparkDl4jMultiLayer(sc.sc(),trainedNetwork);
        
        System.out.println( "\n\n> ---------------------- fit method run! ------------------- \n\n\n" ); 
        
        
        
        // save the model to HDFS: 


//        FileUtil.copyMerge(hdfs, new Path(srcPath), hdfs, new Path(dstPath), false, hadoopConfig, null)        
        
        
     //   FileUtil.fullyDelete(new File( modelPath ));
        
        Path modelPath = new Path( hdfsPathToModelParams );
        
        if ( hdfs.exists( modelPath )) {
        	hdfs.delete( modelPath, true ); 
        } 
        
        OutputStream os = hdfs.create( modelPath );
        
/*            new Progressable() {
                public void progress() {
                    System.out.println("...bytes written: [ "+ this. +" ]");
                } });
  */      
/*        BufferedWriter br = new BufferedWriter( new OutputStreamWriter( os, "UTF-8" ) );
        br.write("Hello World");
        br.close();
        hdfs.close();        
  */      
        
        System.out.println( trainedNetwork.params() );
        
        DataOutputStream dos = new DataOutputStream( os );
        
        //BufferedOutputStream bos = new BufferedOutputStream( os );
        //Nd4j.write( dos,trainedNetwork.params() );
        Nd4j.write( trainedNetwork.params(), dos );
        
      //  FileUtils.write(new File("conf.yaml"),trainedNetwork.conf().toYaml());
      //  System.out.println("F1 = " + precision);
        dos.close();
        hdfs.close();
        
        System.out.println( "Saving model to: " + modelPath );
        
        System.out.println( "\n\n> ---------------------- Saving Model [ done ] ------------------- \n\n\n" );
        


    }	
	
}
