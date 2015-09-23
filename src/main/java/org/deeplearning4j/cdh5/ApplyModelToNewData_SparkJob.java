package org.deeplearning4j.cdh5;

import java.io.BufferedReader;
import java.io.DataInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Arrays;

import org.apache.commons.io.FileUtils;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.evaluation.MulticlassMetrics;
import org.apache.spark.mllib.linalg.DenseVector;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.util.MLUtils;
import org.apache.spark.rdd.RDD;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import scala.Tuple2;
import scala.reflect.ClassManifestFactory;

/**
 * 
 * 1. Load model
 * 
 * 2. Load data to score + cust ID
 * 
 * 3. extract custID, then extract svmLight part from total record
 * 
 * 4. predict class for svmLight record
 * 
 * 5. output: { custID, score }
 * 
 * 
 * @author josh
 *
 */
public class ApplyModelToNewData_SparkJob {

	public static Vector convertSVMLightToVector(String rawLine, int size) {
		
		//Vector sv = Vectors.sparse(3, new int[] {0, 2}, new double[] {1.0, 3.0});
		/*
 SparseVector(int size,
            int[] indices,
            double[] values)
		 */
		
		System.out.println( "line: " + rawLine );
		
		String[] parts = rawLine.trim().split(" ");
		int[] indicies = new int[ parts.length - 1 ];
		double[] values = new double[ parts.length - 1 ];
		
		// skip the label
		for ( int x = 1; x <  parts.length; x++ ) {
			
			String[] indexValueParts = parts[ x ].split(":");
			indicies[ x - 1 ] = (int)Double.parseDouble(indexValueParts[ 0 ]);
			values[ x - 1 ] = Double.parseDouble(indexValueParts[ 1 ]);
			
		}
		
		return Vectors.sparse(size, indicies, values);
		
	}

	public static String getSVMLightRecordFromAeolipileRecord( String aeolipileRecord ) {
    	
    	String work = aeolipileRecord.trim();
    	int firstSpaceIndex = work.indexOf(' ');
    	String newRecord = work.substring(firstSpaceIndex, work.length());
    	
    	return newRecord.trim();

		
	}

	public static String getUniqueIDFromAeolipileRecord( String aeolipileRecord ) {
    	
    	String work = aeolipileRecord.trim();
    	int firstSpaceIndex = work.indexOf(' ');
    	String newRecord = work.substring( 0, firstSpaceIndex );
    	
    	return newRecord.trim();

		
	}
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
        JavaSparkContext sc = new JavaSparkContext(new SparkConf().setAppName("ApplyDL4JModelToNewData"));
        
        String hdfsFilesToEvaluateModel = hdfsPathString; //"hdfs:///user/cloudera/svmlight/*";
        String hdfsPathToModelJSON = args[1];
        String hdfsPathToModelParams = args[2];
        String hdfsPathToOutput = args[3];
        String svmLightFeatureColumns = args[4];
        final int parsedSVMLightFeatureColumns = Integer.parseInt( svmLightFeatureColumns );
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

        
        final SparkDl4jMultiLayer trainedNetworkWrapper = new SparkDl4jMultiLayer(sc.sc(),savedNetwork);
        
        // LOAD the raw text line (Aeolipile format: custID, svmlight record)
        // JavaRDD<ApacheAccessLog> accessLogs = sc.textFile(inputFile)
        //RDD<LabeledPoint> evaluate_svmLight_data_rdd = MLUtils.loadLibSVMFile( sc.sc(), hdfsFilesToEvaluateModel, true, parsedSVMLightFeatureColumns );
        
        JavaRDD<String> rawAeolipileRecords = sc.textFile( hdfsFilesToEvaluateModel ); 
        
        System.out.println( "\n\n\n" +  rawAeolipileRecords.first() + "\n" );
        
        
        // now convert the line into a { custID, svmLight record } pair
        //JavaRDD<LabeledPoint> evaluate_svmLight_data_JavaRDD = JavaRDD.fromRDD( evaluate_svmLight_data_rdd, ClassManifestFactory.fromClass(LabeledPoint.class) );
        
        // { custID, svmLightRecord }
        JavaRDD< Tuple2<String, String> > scoredSVMLightRecords = rawAeolipileRecords.map(new Function<String, Tuple2<String, String>>() {
            @Override
            public Tuple2<String, String> call(String rawRecordString) throws Exception {
                //Vector features = v1.features();
                //Vector normalized = scalarModel.transform(features);
                //return new LabeledPoint( v1.label(), v1.features() );
            	
            	String custID = getUniqueIDFromAeolipileRecord( rawRecordString );
            	String svmLight = getSVMLightRecordFromAeolipileRecord( rawRecordString );
            	
            	
            	Vector svmLightVector = convertSVMLightToVector(svmLight, parsedSVMLightFeatureColumns);
            
            	
                Vector prediction = trainedNetworkWrapper.predict( svmLightVector );
                double max = 0;
                double idx = 0;
                for(int i = 0; i < prediction.size(); i++) {
                    if(prediction.apply(i) > max) {
                        idx = i;
                        max = prediction.apply(i);
                    }
                }

                //return new Tuple2<Object, Object>(idx, p.label());            	
            	
                return new Tuple2<String, String>( custID, (max + "") );
            }
        }).cache();

  //      System.out.println( "Scanned and Cached the dataset " );
        

     
        
        

        /*
        JavaRDD<Tuple2<Object, Object>> newDataScored = evaluate_svmLight_data_JavaRDD.map(
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




//        metricsResult.saveAsTextFile( hdfsPathToEvalOutput );
        
        Path outputPath = new Path( hdfsPathToOutput );
        
        if ( hdfs.exists( outputPath )) {
        	hdfs.delete( outputPath, true ); 
        } 
        
        
        scoredSVMLightRecords.saveAsTextFile( hdfsPathToOutput );
        
        sc.stop();
        
        
        
        
        System.out.println( "New Data Predictions complete..." );    
        
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
