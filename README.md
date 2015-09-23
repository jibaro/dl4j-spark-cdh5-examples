# DL4J Spark Examples for CDH5

Use the latest CDH5.4.2 VM cloudera to try these out:

http://www.cloudera.com/content/cloudera/en/downloads/quickstart_vms/cdh-5-4-x.html

For the general idea of getting spark jobs to run on CDH5:

http://blog.cloudera.com/blog/2014/04/how-to-run-a-simple-apache-spark-app-in-cdh-5/

## Build Model Job

* takes 4 parameters
    * the training dataset already in SVMLight format
    * the model configuration json file
    * the location in HDFS where we'll save the model parameters
    * the number of features total in the SVMLight dataset

### Example Script Usage From Command Line:

spark-submit --class org.deeplearning4j.cdh5.BuildDL4JModel_SVMLight_SparkJob --master yarn /tmp/dl4j-spark-cdh5-examples-1.0-SNAPSHOT.jar hdfs:///user/cloudera/testing/aeolipile/data/train/training_split.txt-r-00000 hdfs:///user/cloudera/testing/aeolipile/models/conf/dlfj_model_conf_20150919_t352pm.json hdfs:///user/cloudera/testing/aeolipile/models/params/dl4j_model_params_dbn_bank_20150921_t1106am.bin 17


## F1 Evaluation Job

Takes the generated model from the model build job and evaluates it.

* takes 5 parameters
    * the test dataset already in SVMLight format
    * the model configuration json file
    * the location in HDFS where we saved the model parameters
    * the location in HDFS where we want to save the evaluation report
    * the number of features total in the SVMLight dataset


### Example Script Usage From Command Line:

spark-submit --class org.deeplearning4j.cdh5.GenerateF1ScoreForModel_SparkJob --master yarn /tmp/dl4j-spark-cdh5-examples-1.0-SNAPSHOT.jar hdfs:///user/cloudera/testing/aeolipile/data/test/test_split.txt-r-00000 hdfs:///user/cloudera/testing/aeolipile/models/conf/dlfj_model_conf_20150919_t352pm.json hdfs:///user/cloudera/testing/aeolipile/models/params/dl4j_model_params_dbn_bank_20150921_t1106am.bin hdfs:///user/cloudera/testing/aeolipile/models/report/f1_bank_20150921_t1221pm 17


## Score New Data Job

An example of how the user would take new data (recordID, svmlight record data) and apply an existing model against it to give: (recordID, modelScore)

* takes 5 parameters
    * the eval dataset already in SVMLight format
    * the model configuration json file
    * the location in HDFS where we saved the model parameters
    * the location in HDFS where we want to save record scores to a text file
    * the number of features total in the SVMLight dataset


### Example Script Usage From Command Line:

spark-submit --class org.deeplearning4j.cdh5.ApplyModelToNewData_SparkJob --master yarn /tmp/dl4j-spark-cdh5-examples-1.0-SNAPSHOT.jar hdfs:///user/cloudera/testing/aeolipile/data/aeolipile_format/aeolipile_format_test.txt hdfs:///user/cloudera/testing/aeolipile/models/conf/dlfj_model_conf_20150919_t352pm.json hdfs:///user/cloudera/testing/aeolipile/models/params/dl4j_model_params_dbn_bank_20150921_t1106am.bin hdfs:///user/cloudera/testing/aeolipile/output/cust20150922 17