package edu.stanford.nlp.jcoref;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.Set;
import java.util.logging.FileHandler;
import java.util.logging.Level;
import java.util.logging.Logger;

import org.ejml.data.Matrix64F;
import org.ejml.simple.SimpleMatrix;

import cs224n.util.Pair;

import sun.java2d.pipe.SpanShapeRenderer.Simple;
import sun.security.krb5.internal.crypto.CksumType;

//import edu.stanford.nlp.classify.LinearRegressor;
import edu.stanford.nlp.io.IOUtils;
import edu.stanford.nlp.jcoref.dcoref.ACEMentionExtractor;
import edu.stanford.nlp.jcoref.dcoref.CoNLLMentionExtractor;
import edu.stanford.nlp.jcoref.dcoref.Constants;
import edu.stanford.nlp.jcoref.dcoref.CorefCluster;
import edu.stanford.nlp.jcoref.dcoref.Dictionaries;
import edu.stanford.nlp.jcoref.dcoref.Document;
import edu.stanford.nlp.jcoref.dcoref.MUCMentionExtractor;
import edu.stanford.nlp.jcoref.dcoref.Mention;
import edu.stanford.nlp.jcoref.dcoref.MentionExtractor;
import edu.stanford.nlp.jcoref.dcoref.SieveCoreferenceSystem;
import edu.stanford.nlp.jcoref.dcoref.SieveCoreferenceSystem.LogFormatter;
import edu.stanford.nlp.jcoref.dcoref.sievepasses.JointArgumentMatch;
import edu.stanford.nlp.jcoref.docclustering.DocumentClustering;
import edu.stanford.nlp.jcoref.docclustering.DocumentClustering.Cluster;
import edu.stanford.nlp.ling.CoreAnnotations.DocIDAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TextAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.parser.lexparser.LexicalizedParser;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.stats.ClassicCounter;
import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.stats.ClassicCounter;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.IntTuple;
import edu.stanford.nlp.util.StringUtils;
import edu.stanford.nlp.util.Triple;

public class RuleBasedJointCorefSystem {
  public static final Logger logger = Logger.getLogger(RuleBasedJointCorefSystem.class.getName());
  private static final String logPath = "C:\\Users\\aman313\\Documents\\study\\cs224n\\Final\\";
  public static final String conllMentionEvalScript = Constants.conllMentionEvalScript;
  private static final boolean doPostProcessing = true;
  //  private static final String serDocs = "/u/heeyoung/corpus-jcb/sample_ser/jcbDocs.ser";
  private static final String jcbSerDocPath = "/u/heeyoung/corpus-jcb/jcb_dev_ser/";
  private static final String conllSerDocPath = "/u/heeyoung/corpus-jcb/conll_train_ser/";
  //private static final String conllSerDocPath = "/u/heeyoung/corpus-jcb/conll_dev_ser/";
  private static String regressorPath = "/u/heeyoung/corpus-jcb/jccModel/";

  SieveCoreferenceSystem corefSystem = null;

  // temp for debug
  public static Counter<String> correctEntityLinkFound = new ClassicCounter<String>();
  public static Counter<String> incorrectEntityLinkFound = new ClassicCounter<String>();
  public static Counter<String> correctEventLinkFound = new ClassicCounter<String>();
  public static Counter<String> incorrectEventLinkFound = new ClassicCounter<String>();

  public static Counter<String> correctEntityLinkMissed = new ClassicCounter<String>();
  public static Counter<String> incorrectEntityLinkMissed = new ClassicCounter<String>();
  public static Counter<String> correctEventLinkMissed = new ClassicCounter<String>();
  public static Counter<String> incorrectEventLinkMissed = new ClassicCounter<String>();

  public static Counter<Double> correctEntityLinksFoundForThreshold = new ClassicCounter<Double>();
  public static Counter<Double> incorrectEntityLinksFoundForThreshold = new ClassicCounter<Double>();
  public static Counter<Double> correctEventLinksFoundForThreshold = new ClassicCounter<Double>();
  public static Counter<Double> incorrectEventLinksFoundForThreshold = new ClassicCounter<Double>();

  public RuleBasedJointCorefSystem(Properties props) throws Exception{
    corefSystem = new SieveCoreferenceSystem(props);
  }
  public static class CoNLLOutputWriter {
    String entityGoldMentionDetectionFile = null;
    String entityGoldCorefFile = null;
    String entityPredictedMentionDetectionFile = null;
    String entityPredictedCorefFile = null;

    String eventGoldMentionDetectionFile = null;
    String eventGoldCorefFile = null;
    String eventPredictedMentionDetectionFile = null;
    String eventPredictedCorefFile = null;

    String bothGoldMentionDetectionFile = null;
    public String bothGoldCorefFile = null;
    String bothPredictedMentionDetectionFile = null;
    public String bothPredictedCorefFile = null;

    PrintWriter entityGoldMentionDetection = null;
    PrintWriter entityGoldCoref = null;
    PrintWriter entityPredictedMentionDetection = null;
    PrintWriter entityPredictedCoref = null;

    PrintWriter eventGoldMentionDetection = null;
    PrintWriter eventGoldCoref = null;
    PrintWriter eventPredictedMentionDetection = null;
    PrintWriter eventPredictedCoref = null;

    PrintWriter bothGoldMentionDetection = null;
    PrintWriter bothGoldCoref = null;
    PrintWriter bothPredictedMentionDetection = null;
    PrintWriter bothPredictedCoref = null;

    public void initialize(String filePrefix) throws FileNotFoundException{
      initialize(filePrefix, false);
    }

    public void initialize(String filePrefix, boolean forOneDoc) throws FileNotFoundException{
      entityGoldMentionDetectionFile = filePrefix+".entity.gold.mentiondetection.txt";
      entityGoldCorefFile = filePrefix+".entity.gold.coref.txt";
      entityPredictedMentionDetectionFile = filePrefix+ ".entity.predicted.mentiondetection.txt";
      entityPredictedCorefFile = filePrefix+ ".entity.predicted.coref.txt";

      eventGoldMentionDetectionFile = filePrefix+".event.gold.mentiondetection.txt";
      eventGoldCorefFile = filePrefix+".event.gold.coref.txt";
      eventPredictedMentionDetectionFile = filePrefix+ ".event.predicted.mentiondetection.txt";
      eventPredictedCorefFile = filePrefix+ ".event.predicted.coref.txt";

      bothGoldMentionDetectionFile = filePrefix+".both.gold.mentiondetection.txt";
      bothGoldCorefFile = filePrefix+".both.gold.coref.txt";
      bothPredictedMentionDetectionFile = filePrefix+ ".both.predicted.mentiondetection.txt";
      bothPredictedCorefFile = filePrefix+ ".both.predicted.coref.txt";

      if(!forOneDoc) {
        logger.info("CONLL ENTITY GOLD MENTION DETECTION FILE: " + entityGoldMentionDetectionFile);
        logger.info("CONLL ENTITY GOLD COREF FILE: " + entityGoldCorefFile);
        logger.info("CONLL ENTITY PREDICTED MENTION DETECTION FILE: " + entityPredictedMentionDetectionFile);
        logger.info("CONLL ENTITY PREDICTED COREF FILE: " + entityPredictedCorefFile);

        logger.info("CONLL EVENT GOLD MENTION DETECTION FILE: " + eventGoldMentionDetectionFile);
        logger.info("CONLL EVENT GOLD COREF FILE: " + eventGoldCorefFile);
        logger.info("CONLL EVENT PREDICTED MENTION DETECTION FILE: " + eventPredictedMentionDetectionFile);
        logger.info("CONLL EVENT PREDICTED COREF FILE: " + eventPredictedCorefFile);

        logger.info("CONLL BOTH GOLD MENTION DETECTION FILE: " + bothGoldMentionDetectionFile);
        logger.info("CONLL BOTH GOLD COREF FILE: " + bothGoldCorefFile);
        logger.info("CONLL BOTH PREDICTED MENTION DETECTION FILE: " + bothPredictedMentionDetectionFile);
        logger.info("CONLL BOTH PREDICTED COREF FILE: " + bothPredictedCorefFile);
      }

      entityGoldMentionDetection = new PrintWriter(new FileOutputStream(entityGoldMentionDetectionFile));
      entityGoldCoref = new PrintWriter(new FileOutputStream(entityGoldCorefFile));
      entityPredictedMentionDetection = new PrintWriter(new FileOutputStream(entityPredictedMentionDetectionFile));
      entityPredictedCoref = new PrintWriter(new FileOutputStream(entityPredictedCorefFile));

      eventGoldMentionDetection = new PrintWriter(new FileOutputStream(eventGoldMentionDetectionFile));
      eventGoldCoref = new PrintWriter(new FileOutputStream(eventGoldCorefFile));
      eventPredictedMentionDetection = new PrintWriter(new FileOutputStream(eventPredictedMentionDetectionFile));
      eventPredictedCoref = new PrintWriter(new FileOutputStream(eventPredictedCorefFile));

      bothGoldMentionDetection = new PrintWriter(new FileOutputStream(bothGoldMentionDetectionFile));
      bothGoldCoref = new PrintWriter(new FileOutputStream(bothGoldCorefFile));
      bothPredictedMentionDetection = new PrintWriter(new FileOutputStream(bothPredictedMentionDetectionFile));
      bothPredictedCoref = new PrintWriter(new FileOutputStream(bothPredictedCorefFile));

      entityGoldMentionDetection.print("#begin document 1\n");
      entityGoldCoref.print("#begin document 1\n");
      entityPredictedMentionDetection.print("#begin document 1\n");
      entityPredictedCoref.print("#begin document 1\n");

      eventGoldMentionDetection.print("#begin document 1\n");
      eventGoldCoref.print("#begin document 1\n");
      eventPredictedMentionDetection.print("#begin document 1\n");
      eventPredictedCoref.print("#begin document 1\n");

      bothGoldMentionDetection.print("#begin document 1\n");
      bothGoldCoref.print("#begin document 1\n");
      bothPredictedMentionDetection.print("#begin document 1\n");
      bothPredictedCoref.print("#begin document 1\n");
    }

    public void closeAll() {
      entityGoldMentionDetection.print("#end document 1");
      entityGoldCoref.print("#end document 1");
      entityPredictedMentionDetection.print("#end document 1");
      entityPredictedCoref.print("#end document 1");

      eventGoldMentionDetection.print("#end document 1");
      eventGoldCoref.print("#end document 1");
      eventPredictedMentionDetection.print("#end document 1");
      eventPredictedCoref.print("#end document 1");

      bothGoldMentionDetection.print("#end document 1");
      bothGoldCoref.print("#end document 1");
      bothPredictedMentionDetection.print("#end document 1");
      bothPredictedCoref.print("#end document 1");

      entityGoldMentionDetection.close();
      entityGoldCoref.close();
      entityPredictedMentionDetection.close();
      entityPredictedCoref.close();

      eventGoldMentionDetection.close();
      eventGoldCoref.close();
      eventPredictedMentionDetection.close();
      eventPredictedCoref.close();

      bothGoldMentionDetection.close();
      bothGoldCoref.close();
      bothPredictedMentionDetection.close();
      bothPredictedCoref.close();
    }
  }
  public static void structuredLearning(Properties props) throws Exception {
    
    String timeStamp = props.getProperty("timeStamp");
    // initialize logger
    FileHandler fh;
    try {
      String logFileName = props.getProperty("jcoref.logFile", "log.txt");
      if(logFileName.endsWith(".txt")) {
        //logFileName = logFileName.substring(0, logFileName.length()-4) +"_"+ timeStamp+".txt";
      } else {
        logFileName = logFileName + "_"+ timeStamp+".txt";
      }
      fh = new FileHandler(logFileName, false);
      logger.addHandler(fh);
      logger.setLevel(Level.FINE);
      fh.setFormatter(new LogFormatter());
    } catch (Exception e) {
      System.err.println("ERROR: cannot initialize logger!");
      throw e;
    }
    
    String trainingData = "/u/heeyoung/corpus-jcb/jcb_dev_train_ser/";
    String tuningData = "/u/heeyoung/corpus-jcb/jcb_dev_tune_ser/";
    
//    String trainingData = "/u/heeyoung/corpus-jcb/sample_train_ser/";
//    String tuningData = "/u/heeyoung/corpus-jcb/sample_tune_ser/";
    
    double[] scores = new double[10];
    RuleBasedJointCorefSystem jcorefSystem;
    for(int t = 0 ; t < 10 ; t++){
      props.setProperty("iteration", String.valueOf(t));
      // training phase
      System.err.println("training iter: "+t);
      props.setProperty("jcoref.trainClassifier", "true");
      props.setProperty("jcoref.readserializedPath", trainingData);
      props.setProperty("jcoref.serializeClassifier", regressorPath+"jccModel.trained."+t+".ser");
      jcorefSystem = new RuleBasedJointCorefSystem(props);
      jcorefSystem.crossDocCorefFromSerialized(props);

      // scoring on tuningData
//      System.err.println("testing on tuningData iter: "+t);
//      props.setProperty("jcoref.trainClassifier", "false");
//      props.setProperty("jcoref.readserializedPath", tuningData);
      props.setProperty("jcoref.jcbSerializedClassifier", regressorPath+"jccModel.trained."+t+".ser");
//      jcorefSystem = new RuleBasedJointCorefSystem(props);
//      scores[t] = jcorefSystem.crossDocCorefFromSerialized(props);
//      
//      logger.fine("t: "+t+ ", score: "+scores[t]);
//      System.out.println("t: "+t+ ", score: "+scores[t]);
    }
    
//    for(int t = 0 ; t < 10 ; t++) {
//      logger.fine("t: "+t+ ", score: "+scores[t]);
//      System.out.println("t: "+t+ ", score: "+scores[t]);
//    }
  }

  public static void main(String[] args) throws Exception {
    //Properties props = StringUtils.argsToProperties(args);
    Properties props = StringUtils.propFileToProperties(args[0]);
    String timeStamp = Calendar.getInstance().getTime().toString().replaceAll("\\s", "-");
    props.setProperty("timeStamp", timeStamp);
    System.err.println("property file has read");
    
    if(Boolean.parseBoolean(props.getProperty("jcoref.srlIndicator"))) JointArgumentMatch.SRL_INDICATOR = true;
    else JointArgumentMatch.SRL_INDICATOR = false;
    
    if(Boolean.parseBoolean(props.getProperty("jcoref.disagree"))) JointArgumentMatch.USE_DISAGREE = true;
    else JointArgumentMatch.USE_DISAGREE = false;
    
    if(Boolean.parseBoolean(props.getProperty("jcoref.userules"))) JointArgumentMatch.USE_RULES = true;
    else JointArgumentMatch.USE_RULES = false;

    if(Boolean.parseBoolean(props.getProperty("jcoref.dopronoun"))) JointArgumentMatch.DOPRONOUN = true;
    else JointArgumentMatch.DOPRONOUN = false;
    
    if(props.containsKey("jcoref.regressorPathToStore")) regressorPath = props.getProperty("jcoref.regressorPathToStore");
        
    if(Boolean.parseBoolean(props.getProperty("jcoref.trainClassifier"))) structuredLearning(props);
    if(props.getProperty("jcoref.readserializedPath").contains("conll")
        && props.getProperty("jcoref.conllSerializedClassifier")==null) props.setProperty("jcoref.trainClassifier", "true");
    if(props.getProperty("jcoref.readserializedPath").contains("conll")) {
      Constants.STRICT_MENTION_BOUNDARY = true;
    }

    // initialize logger
    FileHandler fh;
    try {
      String logFileName = props.getProperty("jcoref.logFile", "log.txt");
      //if(logFileName.endsWith(".txt")) {
        //logFileName = logFileName.substring(0, logFileName.length()-4) +"_"+ timeStamp+".txt";
      //} else {
       // logFileName = logFileName + "_"+ timeStamp+".txt";
      //}
      fh = new FileHandler(logFileName, false);
      logger.addHandler(fh);
      logger.setLevel(Level.FINE);
      fh.setFormatter(new LogFormatter());
    } catch (Exception e) {
      System.err.println("ERROR: cannot initialize logger!");
      throw e;
    }

    RuleBasedJointCorefSystem jcorefSystem = new RuleBasedJointCorefSystem(props);

    boolean READ_SERIALIZED = Boolean.parseBoolean(props.getProperty("jcoref.readserialized", "false"));

    if(READ_SERIALIZED) {
      jcorefSystem.crossDocCorefFromSerialized(props);
    } else {

      LexicalizedParser parser = SieveCoreferenceSystem.makeParser(props); // Load the Stanford Parser
      // MentionExtractor extracts MUC, ACE, CoNLL, or JCB documents
      MentionExtractor mentionExtractor = null;
      if(props.containsKey(Constants.MUC_PROP)){
        mentionExtractor = new MUCMentionExtractor(parser, jcorefSystem.corefSystem.dictionaries, props, jcorefSystem.corefSystem.semantics);
      } else if(props.containsKey(Constants.ACE2004_PROP) || props.containsKey(Constants.ACE2005_PROP)) {
        mentionExtractor = new ACEMentionExtractor(parser, jcorefSystem.corefSystem.dictionaries, props, jcorefSystem.corefSystem.semantics);
      } else if (props.containsKey(Constants.CONLL2011_PROP)) {
        mentionExtractor = new CoNLLMentionExtractor(parser, jcorefSystem.corefSystem.dictionaries, props, jcorefSystem.corefSystem.semantics);

        // run crossDocCoref without document clustering
        jcorefSystem.withinDocCoref(mentionExtractor, props);

      } else {
        JCBMentionExtractor jcbMentionExtractor = new JCBMentionExtractor(new Dictionaries(), props, jcorefSystem.corefSystem.semantics());
        Map<String, JCBDocument> inputs = jcbMentionExtractor.jcbReader.readInputFiles();
        jcorefSystem.crossDocCoref(jcbMentionExtractor, inputs, props);
      }
    }
  }

  public double crossDocCorefFromSerialized(Properties props) throws Exception{
    boolean doScore = (props==null)? false : Boolean.parseBoolean(props.getProperty("jcoref.doScore", "false"));

    String timeStamp = props.getProperty("timeStamp");

    CoNLLOutputWriter conllWriter = null;
    CoNLLOutputWriter conllWriterOneDoc = null;
    String conllOutput = null;

    if(doScore) {
      logger.fine(timeStamp);
      logger.fine(props.toString());
      logger.fine("Thesaurus threshold: top "+Dictionaries.THESAURUS_THRESHOLD+", score > "+Dictionaries.THESAURUS_SCORE_THRESHOLD);
      if(doPostProcessing) logger.fine("do postprocessing");
      logger.fine("use strict mention boundary: "+Constants.STRICT_MENTION_BOUNDARY);

      // prepare conll output
      conllOutput = props.getProperty(Constants.CONLL_OUTPUT_PROP, logPath+"conlloutput\\conlloutput");
      conllWriter = new CoNLLOutputWriter();
     // conllWriter.initialize(conllOutput+"-"+timeStamp+"-iter"+props.getProperty("iteration"));
      conllWriter.initialize(conllOutput+"-iter"+props.getProperty("iteration"));
    }
    String serializedCorpus = props.getProperty("jcoref.readserializedPath", jcbSerDocPath);
    /*if(corefSystem.jcc.regressor!=null) {
      logger.fine("Regressor features: ");
      logger.fine(((LinearRegressor<String>)corefSystem.jcc.regressor).getTopFeatures(100).toString());
    }
*/
    for(File file : IOUtils.iterFilesRecursive(new File(serializedCorpus), "ser")) {
      JointCorefDocument jDoc = IOUtils.readObjectFromFile(file);

//      jDoc.docForTracing = IOUtils.readObjectFromFile(file);

      logger.fine("process Doc "+jDoc.docID+" -> "+jDoc.originalDocs);

      if(doScore) {
        jDoc.extractGoldCorefClusters();
        jDoc.goldMentionCounterInCluster = new ClassicCounter<Integer>();
        for(CorefCluster c : jDoc.goldCorefClusters.values()) {
          jDoc.goldMentionCounterInCluster.incrementCount(c.getClusterID(), c.getCorefMentions().size());
        }
        printRawDoc(jDoc, true);
        printRawDoc(jDoc, false);
        // print conll output before coref
        printConllOutput(jDoc, conllWriter, false);
        conllWriterOneDoc = new CoNLLOutputWriter();
       // conllWriterOneDoc.initialize(conllOutput+timeStamp+"-iter"+props.getProperty("iteration")+"-temp", true);
        conllWriterOneDoc.initialize(conllOutput+"-iter"+props.getProperty("iteration")+"-temp", true);

        printConllOutput(jDoc, conllWriterOneDoc, false);
      }

      // preprocessing
      // TODO

//      printDocInfo(jDoc);

      // coref
      this.corefSystem.coref(jDoc);

      // postprocessing
      // TODO
      if(doPostProcessing) postProcessing(jDoc);
      
      
      spectralCoCluster(jDoc); // maybe this needs to be done before the postprocessing
      
      

      logger.fine("Doc "+jDoc.docID+" processed");

      if(doScore) {
        this.corefSystem.printTopK(logger, jDoc, this.corefSystem.semantics);
        printConllOutput(jDoc, conllWriterOneDoc, true);
        printConllOutput(jDoc, conllWriter, true);
        conllWriterOneDoc.closeAll();
        if(!props.getProperty("jcoref.readserializedPath").contains("conll")) {
          logger.fine("SCORE: this doc -----------------------------------------------");
          printSummary(conllWriterOneDoc);
          logger.fine("SCORE: accumulated ------------------------------------------------");
          printSummary(conllWriter);
        }

        logger.fine("-------- Argument Precision Counts ------------------------------");
        logger.fine("correctEntityLinkFound: ");
        for(String s : correctEntityLinkFound.keySet()) {
          logger.fine("\t\t"+s+" -> "+correctEntityLinkFound.getCount(s));
        }
        logger.fine("incorrectEntityLinkFound: ");
        for(String s : incorrectEntityLinkFound.keySet()) {
          logger.fine("\t\t"+s+" -> "+incorrectEntityLinkFound.getCount(s));
        }
        logger.fine("correctEventLinkFound: ");
        for(String s : correctEventLinkFound.keySet()) {
          logger.fine("\t\t"+s+" -> "+correctEventLinkFound.getCount(s));
        }
        logger.fine("incorrectEventLinkFound: ");
        for(String s : incorrectEventLinkFound.keySet()) {
          logger.fine("\t\t"+s+" -> "+incorrectEventLinkFound.getCount(s));
        }
        logger.fine("correctEntityLinkMissed: ");
        for(String s : correctEntityLinkMissed.keySet()) {
          logger.fine("\t\t"+s+" -> "+correctEntityLinkMissed.getCount(s));
        }
        logger.fine("incorrectEntityLinkMissed: ");
        for(String s : incorrectEntityLinkMissed.keySet()) {
          logger.fine("\t\t"+s+" -> "+incorrectEntityLinkMissed.getCount(s));
        }
        logger.fine("correctEventLinkMissed: ");
        for(String s : correctEventLinkMissed.keySet()) {
          logger.fine("\t\t"+s+" -> "+correctEventLinkMissed.getCount(s));
        }
        logger.fine("incorrectEventLinkMissed: ");
        for(String s : incorrectEventLinkMissed.keySet()) {
          logger.fine("\t\t"+s+" -> "+incorrectEventLinkMissed.getCount(s));
        }
        logger.fine("-----------------------------------------------------------------");
        logger.fine("Links Precision for each threshold value");
        logger.fine("correctEntityLinksFoundForThreshold: ");
        for(Double d : correctEntityLinksFoundForThreshold.keySet()) {
          logger.fine("\t\t"+d+" -> "+correctEntityLinksFoundForThreshold.getCount(d));
        }
        logger.fine("incorrectEntityLinksFoundForThreshold: ");
        for(Double d : incorrectEntityLinksFoundForThreshold.keySet()) {
          logger.fine("\t\t"+d+" -> "+incorrectEntityLinksFoundForThreshold.getCount(d));
        }
        logger.fine("correctEventLinksFoundForThreshold: ");
        for(Double d : correctEventLinksFoundForThreshold.keySet()) {
          logger.fine("\t\t"+d+" -> "+correctEventLinksFoundForThreshold.getCount(d));
        }
        logger.fine("incorrectEventLinksFoundForThreshold: ");
        for(Double d : incorrectEventLinksFoundForThreshold.keySet()) {
          logger.fine("\t\t"+d+" -> "+incorrectEventLinksFoundForThreshold.getCount(d));
        }
        logger.fine("");

        logger.fine(" Mention : Cluster ratio ");
        logger.fine("\tGold Mention Size: "+jDoc.allGoldMentions.size());
        logger.fine("\tGold Cluster Size: "+jDoc.goldCorefClusters.size());
        logger.fine("\tPredicted Mention Size: "+jDoc.allPredictedMentions.size());
        logger.fine("\tPredicted Cluster Size: "+jDoc.corefClusters.size());
        logger.fine("");

      }
      logger.fine(SieveCoreferenceSystem.sameLemmaPredicates.toString());
    }
    
    /*if( SieveCoreferenceSystem.trainJointCorefClassifier) {
      corefSystem.jcc.train(Double.parseDouble(props.getProperty("jcoref.oldCoefficientWeight", "0.7")));
      corefSystem.jcc.serialize(props.getProperty("jcoref.serializeClassifier"));
      System.err.println();
    }*/

    double score = -1;
    if(doScore) {
      conllWriter.closeAll();
      logger.fine("SCORE: final -----------------------------------------------");
      printSummary(conllWriter);
      String bcub = SieveCoreferenceSystem.getConllEvalSummary(conllMentionEvalScript, conllWriter.bothGoldCorefFile, conllWriter.bothPredictedCorefFile, "bcub");
      bcub = bcub.split("F1: ")[2].split("%")[0];
      score = Double.parseDouble(bcub);
      logger.info("done");
    }
    return score;
  }

 public void spectralCoCluster(JointCorefDocument doc){
	 
	 /*
	  * Spectral Constraint Modeling algorithm 
	  */
	 
	 // Construct the Bipartite edge matrix E and Constraint matrix C
	 //constructBandC(doc);
	 ArrayList<SimpleMatrix> matrices= constructEandC(doc);
	 
	 // We have the input matrices. Now time to do the clustering
	 double delta = 1.0; // TODO : Take this in as a parameter 
	 int k = 5;
	 int l = 5; // need to set these smarter 
	 ArrayList<SimpleMatrix> partitionMatrices = scm(matrices.get(0),matrices.get(1),delta,k,l);
	 
	 // Now process the partitionMatrices to get the new clustering
	 
	 Map<Integer,CorefCluster>  finalClusters = getClustering(partitionMatrices);
	 
 }
 
 public Map<Integer, CorefCluster> getClustering(ArrayList<SimpleMatrix> partitionMatrices){
	 
	 return null;
 }
 
 public ArrayList<SimpleMatrix> scm(SimpleMatrix E, SimpleMatrix C, double confPar, int k, int l){
	 
	 return null;
 }
 
 // We make this global in order to map back from the matrix index to the mentions
 // Note that we shall structure the matrix such that verb mentions are in rows and other mention in cols 
 // in case that the matrix dimention contains both the verb and the other mentions then the other mentions appear afte the verbs
 // the order within both kind of mentions is preseved to be the same as these lists 
 ArrayList<Mention> verbMentions = new ArrayList<Mention>();
 ArrayList<Mention> otherMentions = new ArrayList<Mention>(); // This is the list of other mentions; Mostly should be nominal and pronominal
 
 Map<Integer,Mention> verbMentionMap = new HashMap<Integer, Mention>();
 Map<Integer,Mention> otherMentionMap = new HashMap<Integer, Mention>();
 
 
 ArrayList<SimpleMatrix> constructEandC(JointCorefDocument doc){
	 
	 //Map<Integer,CorefCluster>  clusters = doc.corefClusters;
	 /*
	  * Go through all the clusters 
	  *   for each mention in a cluster. 
	  *   	if the mention is of type noun 
	  *   		increase noun count
	  *   		
	  *   	if the mention is of type verb
	  */
	 
	 //Set<Integer> clusterIds = clusters.keySet();
	 int id;
	 //verbMentions.clear();
	 //otherMentions.clear();
	 verbMentionMap.clear();
	 otherMentionMap.clear();
	 
	 int numVerbs =0;
	 int numOthers =0;
	/* Iterator<Integer> clusterIdsIterator = clusterIds.iterator();
	 while( clusterIdsIterator.hasNext()){
		 id =  clusterIdsIterator.next();
		 CorefCluster cluster = clusters.get(id);
		 Iterator<Mention> clusterIterator = cluster.getCorefMentions().iterator();
		 while(clusterIterator.hasNext()){
			 Mention m = clusterIterator.next();
			 if(m.isVerb){
				 verbMentions.add(m);
				 numVerbs++;
			 }
			 else{
				 otherMentions.add(m);
				 numOthers++;
			 }
			 
		 }
		 
	 }*/
	 
	 Iterator<List<Mention>> listIterator = doc.getOrderedMentions().iterator();
	 while(listIterator.hasNext()){
		 Iterator< Mention> mentionIterator = listIterator.next().iterator();
		 while(mentionIterator.hasNext()){
			 Mention m = mentionIterator.next();
			 if(m.isVerb){
				 verbMentionMap.put(numVerbs, m);
				 numVerbs++;
			 }
			 else{
				 otherMentionMap.put(numOthers, m);
				 numOthers++;
			 }
		 }
		 
	 }
	 
	 
	 // Mentions have been sorted into two lists 
	 // Now we can initialize the matrices 
	 int numVerbMentions = verbMentionMap.size();
	 int numOtherMentions = otherMentionMap.size();
	 
	 SimpleMatrix E = new SimpleMatrix(numVerbMentions, numOtherMentions); 
	 SimpleMatrix C = new SimpleMatrix(numOtherMentions+numVerbMentions, numOtherMentions+numVerbMentions);
	 
	 
	 // Now we need to populate these matrices 
	 // We do this the dumb way for now 
	 int verbIndex = 0;
	 int otherIndex =0;
	 Iterator<Integer> verbMentions = verbMentionMap.keySet().iterator();
	 while(verbMentions.hasNext()){
		 int verbMentionCode = verbMentions.next();
		 Mention verbMention = verbMentionMap.get(verbMentionCode);
		 Iterator<Integer> otherMentions = otherMentionMap.keySet().iterator();
		 verbIndex =  verbMentionCode;
		 while(otherMentions.hasNext()){
			 int otherMentionCode = otherMentions.next();
			 Mention otherMention = otherMentionMap.get(otherMentionCode);
			 // Check if the other mention is an argument of the verbmention. If yes set the Eij
			 otherIndex =  otherMentionCode;
			 E.set(verbIndex, otherIndex, 0.0);
			 double weight = argLikelihood(verbMention.srlArgs,otherMention);
			 if( weight> 0.2){ // If the likehood is not too low, use this score as an ed
				 
				 E.set(verbIndex, otherIndex, weight);
			 }
			 if(verbMention.corefClusterID == otherMention.corefClusterID){
				 C.set(verbIndex, otherIndex, 1);

			 }
			 
			 otherIndex++;
		 }
		 		 
		 verbIndex++;
	 } // E matrix constructed
	  
	 //Now construct the C matrix 
	 // This is a square matrix and each 
	 verbMentions = verbMentionMap.keySet().iterator();
	 while(verbMentions.hasNext()){
		 int verbMentionCode = verbMentions.next();
		 Mention verbMention = verbMentionMap.get(verbMentionCode);
		 Iterator<Integer> otherMentions = otherMentionMap.keySet().iterator();
		 verbIndex =  verbMentionCode;
	 
		 Iterator<Integer> verbMentions1 = verbMentionMap.keySet().iterator();
		 while(verbMentions1.hasNext()){
			
			 int verbMentionCode1 = verbMentions1.next();
			 Mention verbMention1 = verbMentionMap.get(verbMentionCode1);
			 int verbIndex1 =  verbMentionCode1;
			 if(verbMention.corefClusterID == verbMention1.corefClusterID){
				 C.set(verbIndex, verbIndex1, 1);
			 }
	 
		 }
	}	 
	 
	 Iterator<Integer> otherMentions = otherMentionMap.keySet().iterator();
	 while(otherMentions.hasNext()){
		 int otherMentionCode = otherMentions.next();
		 Mention otherMention = otherMentionMap.get(otherMentionCode);
		 // Check if the other mention is an argument of the verbmention. If yes set the Eij
		 otherIndex =  otherMentionCode;
		 Iterator<Integer> otherMentions1 = otherMentionMap.keySet().iterator();
		 while(otherMentions1.hasNext()){
			 int otherMentionCode1 = otherMentions1.next();
			 Mention otherMention1 = otherMentionMap.get(otherMentionCode1);
			 // Check if the other mention is an argument of the verbmention. If yes set the Eij
			 int otherIndex1 =  otherMentionCode1;
			 if(otherMention.corefClusterID == otherMention1.corefClusterID){
				 C.set(otherIndex, otherIndex1, 1);				 
			 }
			 
		 }
	 }
	 
	 ArrayList<SimpleMatrix> matrices = new ArrayList<SimpleMatrix>();
	 matrices.add(E);
	 matrices.add(C);
	 
	 return matrices;
 }
 
 public double argLikelihood(Map<String,Mention> srl, Mention other){
	 double score = 0.0;
	 double totalscore = 3;
	 
	 if(srl.get("A0")!=null && srl.get("A0").headString.equalsIgnoreCase(other.headString)){
		 score = score +1;
	 }
	 
	 if(srl.get("A1")!=null && srl.get("A1").headString.equalsIgnoreCase(other.headString)){
		 score = score+1;
	 }
	 
	 
	 if(srl.get("A2")!=null && srl.get("A2").headString.equalsIgnoreCase(other.headString)){
		 score = score+1;
	 }

	 if(srl.get("A0")!=null && srl.get("A0").gendersAgree(other)){
		 score = score +0.5;
	 }
	 
	 if(srl.get("A1")!=null && srl.get("A1").gendersAgree(other)){
		 score = score +0.5;
	 }

	 return score/totalscore;
 }
  
  public void withinDocCoref(MentionExtractor mentionExtractor, Properties props) throws Exception {
    boolean STORE_SERIALZIED = Boolean.parseBoolean(props.getProperty("jcoref.storeserialized", "false"));

    boolean doScore = (props==null)? false : Boolean.parseBoolean(props.getProperty("jcoref.doScore", "false"));

    String timeStamp = Calendar.getInstance().getTime().toString().replaceAll("\\s", "-");

    CoNLLOutputWriter conllWriter = null;
    CoNLLOutputWriter conllWriterOneDoc = null;
    String conllOutput = null;

    if(doScore) {
      // initialize logger
      FileHandler fh;
      try {
        String logFileName = props.getProperty("jcoref.logFile", "log.txt");
        if(logFileName.endsWith(".txt")) {
          logFileName = logFileName.substring(0, logFileName.length()-4) +"_"+ timeStamp+".txt";
        } else {
          logFileName = logFileName + "_"+ timeStamp+".txt";
        }
        fh = new FileHandler(logFileName, false);
        logger.addHandler(fh);
        logger.setLevel(Level.FINE);
        fh.setFormatter(new LogFormatter());
      } catch (Exception e) {
        System.err.println("ERROR: cannot initialize logger!");
        throw e;
      }

      logger.fine(timeStamp);
      logger.fine(props.toString());
      logger.fine("Thesaurus threshold: top "+Dictionaries.THESAURUS_THRESHOLD+", score > "+Dictionaries.THESAURUS_SCORE_THRESHOLD);
      if(doPostProcessing) logger.fine("do postprocessing");
      logger.fine("use strict mention boundary: "+Constants.STRICT_MENTION_BOUNDARY);

      // prepare conll output
      conllOutput = props.getProperty(Constants.CONLL_OUTPUT_PROP, logPath+"/conlloutput/conlloutput");
      conllWriter = new CoNLLOutputWriter();
      conllWriter.initialize(conllOutput+"-"+timeStamp);

    }

    // for each document, run mention detection and coref
    JointCorefDocument jDoc = null;
    Document doc = null;
    while((doc = mentionExtractor.nextDoc())!=null) {
      jDoc = mentionExtractor.extractJointCorefDocument(doc);
      System.err.println();

      if(STORE_SERIALZIED) IOUtils.writeObjectToFile(jDoc, conllSerDocPath+jDoc.docID+".ser");

      if(!STORE_SERIALZIED) {
        printDocInfo(jDoc);

        // coref
        this.corefSystem.coref(jDoc);

        // postprocessing
        // TODO
        if(doPostProcessing) postProcessing(jDoc);
      }
      logger.fine("Doc "+jDoc.docID+" processed");
    }

    if(doScore) {
      conllWriter.closeAll();
      logger.fine("SCORE: final -----------------------------------------------");
      printSummary(conllWriter);
      logger.info("done");
    }
  }

  public void crossDocCoref(JCBMentionExtractor jcbMentionExtractor, Map<String, JCBDocument> inputs, Properties props) throws Exception{
    boolean STORE_SERIALZIED = Boolean.parseBoolean(props.getProperty("jcoref.storeserialized", "false"));

    boolean doScore = (props==null)? false : Boolean.parseBoolean(props.getProperty("jcoref.doScore", "false"));

    String timeStamp = Calendar.getInstance().getTime().toString().replaceAll("\\s", "-");

    CoNLLOutputWriter conllWriter = null;
    CoNLLOutputWriter conllWriterOneDoc = null;
    String conllOutput = null;

    if(doScore) {
      // initialize logger
      FileHandler fh;
      try {
        String logFileName = props.getProperty("jcoref.logFile", "log.txt");
        if(logFileName.endsWith(".txt")) {
          logFileName = logFileName.substring(0, logFileName.length()-4) +"_"+ timeStamp+".txt";
        } else {
          logFileName = logFileName + "_"+ timeStamp+".txt";
        }
        fh = new FileHandler(logFileName, false);
        logger.addHandler(fh);
        logger.setLevel(Level.FINE);
        fh.setFormatter(new LogFormatter());
      } catch (Exception e) {
        System.err.println("ERROR: cannot initialize logger!");
        throw e;
      }

      logger.fine(timeStamp);
      logger.fine(props.toString());
      logger.fine("Thesaurus threshold: top "+Dictionaries.THESAURUS_THRESHOLD+", score > "+Dictionaries.THESAURUS_SCORE_THRESHOLD);
      if(doPostProcessing) logger.fine("do postprocessing");
      logger.fine("Annotated sentence only? "+jcbMentionExtractor.detectMentionsInAnnotatedSentencesOnly);
      logger.fine("use strict mention boundary: "+Constants.STRICT_MENTION_BOUNDARY);

      // prepare conll output
      conllOutput = props.getProperty(Constants.CONLL_OUTPUT_PROP, logPath+"/conlloutput/conlloutput");
      conllWriter = new CoNLLOutputWriter();
      conllWriter.initialize(conllOutput+"-"+timeStamp);

    }

    // document clustering
    boolean USE_GOLD_CLUSTERING = Boolean.parseBoolean(props.getProperty("jcoref.goldclustering", "false"));
    logger.info("USE GOLD CLUSTERING? "+USE_GOLD_CLUSTERING);
    Map<Integer, Cluster> clusters;
    if(USE_GOLD_CLUSTERING) clusters = DocumentClustering.getGoldDocumentClusters(inputs);
    else clusters = DocumentClustering.getDocumentClusters(inputs);

    // build meta doc for coref
    Map<String, JCBDocument> metaDocs = buildMetaDocs(inputs, clusters);

    // for each document, run mention detection and coref
    //    List<JointCorefDocument> jDocs = new ArrayList<JointCorefDocument>();
    JointCorefDocument jDoc = null;
    for(JCBDocument metaDoc : metaDocs.values()) {
      logger.fine("process Doc "+metaDoc.docID+" -> "+metaDoc.docsInMeta);

      // mention detection
      jDoc = jcbMentionExtractor.extractMentions(metaDoc);

      if(doScore) {
        jDoc.extractGoldCorefClusters();
        jDoc.goldMentionCounterInCluster = new ClassicCounter<Integer>();
        for(CorefCluster c : jDoc.goldCorefClusters.values()) {
          jDoc.goldMentionCounterInCluster.incrementCount(c.getClusterID(), c.getCorefMentions().size());
        }
        printRawDoc(jDoc, true);
        printRawDoc(jDoc, false);
        // print conll output before coref
        printConllOutput(jDoc, conllWriter, false);
        conllWriterOneDoc = new CoNLLOutputWriter();
        conllWriterOneDoc.initialize(conllOutput+timeStamp+"-temp");
        printConllOutput(jDoc, conllWriterOneDoc, false);
      }
      // preprocessing
      // TODO

      if(STORE_SERIALZIED) IOUtils.writeObjectToFile(jDoc, jcbSerDocPath+jDoc.docID+".ser");

      if(!STORE_SERIALZIED) {
        printDocInfo(jDoc);

        // coref
        this.corefSystem.coref(jDoc);

        // postprocessing
        // TODO
        if(doPostProcessing) postProcessing(jDoc);
      }

      //      jDocs.add(jDoc);
      logger.fine("Doc "+jDoc.docID+" processed");

      if(doScore) {
        this.corefSystem.printTopK(logger, jDoc, this.corefSystem.semantics);
        printConllOutput(jDoc, conllWriterOneDoc, true);
        printConllOutput(jDoc, conllWriter, true);
        conllWriterOneDoc.closeAll();
        logger.fine("SCORE: this doc -----------------------------------------------");
        printSummary(conllWriterOneDoc);
        logger.fine("SCORE: accumulated ------------------------------------------------");
        printSummary(conllWriter);
      }
    }

    if(doScore) {
      conllWriter.closeAll();
      logger.fine("SCORE: final -----------------------------------------------");
      printSummary(conllWriter);
      logger.info("done");
    }
    //    if(STORE_SERIALZIED) IOUtils.writeObjectToFile(jDocs, serDocPath);

    //    return jDocs;
  }

  private static Map<String, JCBDocument> buildMetaDocs(Map<String, JCBDocument> inputs, Map<Integer, Cluster> clusters) {
    Map<String, JCBDocument> metaDocs = new HashMap<String, JCBDocument>();

    for(int cID : clusters.keySet()){
      Cluster c = clusters.get(cID);
      String metaDocID = "Meta"+cID;
      List<JCBDocument> jcbDocs = new ArrayList<JCBDocument>();

      for(DocumentClustering.Document d : c.docs){
        jcbDocs.add(inputs.get(d.docID));
      }
      JCBDocument metaDoc = new JCBDocument(metaDocID, jcbDocs);
      metaDocs.put(metaDocID, metaDoc);
    }
    return metaDocs;
  }

  private static void postProcessing(JointCorefDocument document) {
    Set<IntTuple> removeSet = new HashSet<IntTuple>();
    Set<Integer> removeClusterSet = new HashSet<Integer>();

    for(CorefCluster c : document.corefClusters.values()){
      Set<Mention> removeMentions = new HashSet<Mention>();
      for(Mention m : c.getCorefMentions()) {
        // remove twinless mention
        //        if(m.twinless) {
        //          removeMentions.add(m);
        //          removeSet.add(document.positions.get(m));
        //          m.corefClusterID = m.mentionID;
        //          continue;
        //        }
        //        if(m.isEvent) continue; // don't remove singleton event

        if((Constants.REMOVE_APPOSITION_PREDICATENOMINATIVES
            && ((m.appositions!=null && m.appositions.size() > 0)
                || (m.predicateNominatives!=null && m.predicateNominatives.size() > 0)
                || (m.relativePronouns!=null && m.relativePronouns.size() > 0)))){
          removeMentions.add(m);
          removeSet.add(document.positions.get(m));
          m.corefClusterID = -1;
        }

        // remove mentions in the sentences which have no gold mentions
        if(document.goldOrderedMentionsBySentence.get(m.sentNum).size()==0) {
          removeMentions.add(m);
          removeSet.add(document.positions.get(m));
          m.corefClusterID = -1;
        }
      }
      c.getCorefMentions().removeAll(removeMentions);
      if(Constants.REMOVE_SINGLETONS && c.getCorefMentions().size() < 2) {
        removeClusterSet.add(c.getClusterID());
      }
    }
    for(int removeId : removeClusterSet){
      document.corefClusters.remove(removeId);
    }
    for(IntTuple pos : removeSet){
      document.positions.remove(pos);
    }
  }

  public static void printConllOutput(JointCorefDocument jDoc, CoNLLOutputWriter conllWriter, boolean afterCoref) {
    if(afterCoref){
      printConllOutput(jDoc, conllWriter.entityGoldCoref, false, false, true, true, Constants.STRICT_MENTION_BOUNDARY);
      printConllOutput(jDoc, conllWriter.eventGoldCoref, false, true, true, true, Constants.STRICT_MENTION_BOUNDARY);
      printConllOutput(jDoc, conllWriter.bothGoldCoref, true, true, true, true, Constants.STRICT_MENTION_BOUNDARY);

      printConllOutput(jDoc, conllWriter.entityPredictedCoref, false, false, false, true, Constants.STRICT_MENTION_BOUNDARY);
      printConllOutput(jDoc, conllWriter.eventPredictedCoref, false, true, false, true, Constants.STRICT_MENTION_BOUNDARY);
      printConllOutput(jDoc, conllWriter.bothPredictedCoref, true, true, false, true, Constants.STRICT_MENTION_BOUNDARY);

    } else{
      printConllOutput(jDoc, conllWriter.entityGoldMentionDetection, false, false, true, false, Constants.STRICT_MENTION_BOUNDARY);
      printConllOutput(jDoc, conllWriter.eventGoldMentionDetection, false, true, true, false, Constants.STRICT_MENTION_BOUNDARY);
      printConllOutput(jDoc, conllWriter.bothGoldMentionDetection, true, false, true, false, Constants.STRICT_MENTION_BOUNDARY);

      printConllOutput(jDoc, conllWriter.entityPredictedMentionDetection, false, false, false, false, Constants.STRICT_MENTION_BOUNDARY);
      printConllOutput(jDoc, conllWriter.eventPredictedMentionDetection, false, true, false, false, Constants.STRICT_MENTION_BOUNDARY);
      printConllOutput(jDoc, conllWriter.bothPredictedMentionDetection, true, false, false, false, Constants.STRICT_MENTION_BOUNDARY);
    }
  }

  public static void printConllOutput(
      JointCorefDocument document,
      PrintWriter writer,
      boolean printBoth, boolean isEvent, boolean isGold, boolean filterSingletons, boolean strictBoundary) {

    List<List<Mention>> orderedMentions = (isGold)? document.goldOrderedMentionsBySentence : document.predictedOrderedMentionsBySentence;

    Annotation anno = document.annotation;
    StringBuilder sb = new StringBuilder();

    // print meta doc info
    if(document.originalDocs!=null) {
      sb.append("##### META DOC "+document.docID +": ");
      for(Triple<String, Integer, Integer> doc : document.originalDocs) {
        sb.append(doc.first()).append(", ");
      }
      sb.append("\n");
    } else {
      sb.append("#begin document ").append(document.docID).append("\n");
    }

    String oldDocID = "";
    List<CoreMap> sentences = anno.get(SentencesAnnotation.class);
    for(int sentNum = 0 ; sentNum < sentences.size() ; sentNum++){
      List<CoreLabel> sentence = sentences.get(sentNum).get(TokensAnnotation.class);
      Map<Integer,Set<Mention>> mentionBeginOnly = new HashMap<Integer,Set<Mention>>();
      Map<Integer,Set<Mention>> mentionEndOnly = new HashMap<Integer,Set<Mention>>();
      Map<Integer,Set<Mention>> mentionBeginEnd = new HashMap<Integer,Set<Mention>>();

      for(int i=0 ; i<sentence.size(); i++){
        mentionBeginOnly.put(i, new LinkedHashSet<Mention>());
        mentionEndOnly.put(i, new LinkedHashSet<Mention>());
        mentionBeginEnd.put(i, new LinkedHashSet<Mention>());
      }
      boolean isAnnotatedSentence = (document.goldOrderedMentionsBySentence.get(sentNum).size() > 0);

      for(Mention m : orderedMentions.get(sentNum)) {
        
        // when we use ontonotes: no event mentions
        if(!document.docID.startsWith("Meta") && (m.isVerb || m.isEvent)) {
          continue;
        }

        boolean corefWithGoldEventMention = false;
        boolean corefWithGoldEntityMention = false;

        if(isGold) {
          if(!printBoth && m.isEvent != isEvent) continue;
          if(filterSingletons && document.goldMentionCounterInCluster.getCount(m.goldCorefClusterID) < 2) continue;
        } else {

          // this is done in post processing now
          //          if(!isAnnotatedSentence) {
          //            continue;
          //          }

          // after coref
          if(filterSingletons) {
            if(!document.corefClusters.containsKey(m.corefClusterID)
                || document.corefClusters.get(m.corefClusterID).getCorefMentions().size() < 2) continue;
            if(!printBoth) {
              for(Mention corefMention : document.corefClusters.get(m.corefClusterID).getCorefMentions()) {
                if(!corefMention.twinless){
                  if(document.allGoldMentions.get(corefMention.mentionID).isEvent) corefWithGoldEventMention = true;
                  else corefWithGoldEntityMention = true;
                }
              }
              // if there is at least one mention coreferent with gold entity or event mention to score, then don't skip.
              // if there is no such mention, and the cluster is known as another kind (event cluster for entity scoring or vice versa), then skip the cluster.
              // if all mentions in the cluster are spurious, don't skip.
              if(isEvent && !corefWithGoldEventMention && corefWithGoldEntityMention) continue;
              if(!isEvent && corefWithGoldEventMention && !corefWithGoldEntityMention) continue;
            }
          }
        }

        Mention m2 = m;

        // comment to see annotation agreement
        if(!strictBoundary && isGold && !m.twinless) m2 = document.allPredictedMentions.get(m.mentionID);
        int endIndex = (m2.additionalSpanEndIndex == -1)? m2.endIndex : m2.additionalSpanEndIndex;
        if(m2.startIndex==endIndex-1) {
          mentionBeginEnd.get(m2.startIndex).add(m);
        } else {
          mentionBeginOnly.get(m2.startIndex).add(m);
          mentionEndOnly.get(endIndex-1).add(m);
        }
      }

      String docID = sentences.get(sentNum).get(DocIDAnnotation.class);
      if(docID != oldDocID) {
        oldDocID = docID;
        sb.append("# DOC ").append(docID).append("\n");
      }

      for(int i=0 ; i<sentence.size(); i++){
        StringBuilder sb2 = new StringBuilder();
        for(Mention m : mentionBeginOnly.get(i)){
          if (sb2.length() > 0) {
            sb2.append("|");
          }
          int corefClusterId = (isGold)? m.goldCorefClusterID:m.corefClusterID;
          sb2.append("(").append(corefClusterId);
        }
        for(Mention m : mentionBeginEnd.get(i)){
          if (sb2.length() > 0) {
            sb2.append("|");
          }
          int corefClusterId = (isGold)? m.goldCorefClusterID:m.corefClusterID;
          sb2.append("(").append(corefClusterId).append(")");
        }
        for(Mention m : mentionEndOnly.get(i)){
          if (sb2.length() > 0) {
            sb2.append("|");
          }
          int corefClusterId = (isGold)? m.goldCorefClusterID:m.corefClusterID;
          sb2.append(corefClusterId).append(")");
        }
        if(sb2.length() == 0) sb2.append("-");

        //        String[] columns = conllSentence.get(i);
        //        for(int j = 0 ; j < columns.length-1 ; j++){
        //          String column = columns[j];
        //          sb.append(column).append("\t");
        //        }
        sb.append(i+1).append("\t").append(sentence.get(i).get(TextAnnotation.class)).append("\t");
        sb.append(sb2).append("\n");
      }
      sb.append("\n");
    }

    //    sb.append("#end document").append("\n");
    //    sb.append("#end document ").append(document.docID).append("\n");

    if(document.originalDocs==null) {
      sb.append("#end document ").append(document.docID).append("\n");
    }

    writer.print(sb.toString());
    writer.flush();
  }

  private static void printRawDoc(JointCorefDocument document, boolean gold) throws FileNotFoundException {
    List<CoreMap> sentences = document.annotation.get(SentencesAnnotation.class);
    List<List<Mention>> allMentions;
    StringBuilder doc = new StringBuilder();

    if(gold) {
      allMentions = document.goldOrderedMentionsBySentence;
      doc.append("New TOPIC ").append(document.docID).append(": (GOLD MENTIONS) ==================================================\n");
    } else {
      allMentions = document.predictedOrderedMentionsBySentence;
      doc.append("New TOPIC ").append(document.docID).append(": (Predicted Mentions) ==================================================\n");
    }

    Counter<Integer> mentionCount = new ClassicCounter<Integer>();
    for(List<Mention> l : allMentions) {
      for(Mention m : l){
        mentionCount.incrementCount(m.goldCorefClusterID);
      }
    }

    for(int i = 0 ; i<sentences.size(); i++) {
      CoreMap sentence = sentences.get(i);
      List<Mention> mentions = allMentions.get(i);

      String[] tokens = sentence.get(TextAnnotation.class).split(" ");
      Counter<Integer> startCounts = new ClassicCounter<Integer>();
      Counter<Integer> endCounts = new ClassicCounter<Integer>();
      HashMap<Integer, Set<Mention>> endMentions = new HashMap<Integer, Set<Mention>>();
      for (Mention m : mentions) {
        if(m.isEvent) continue;
        startCounts.incrementCount(m.startIndex);
        endCounts.incrementCount(m.endIndex);
        if(!endMentions.containsKey(m.endIndex)) endMentions.put(m.endIndex, new HashSet<Mention>());
        endMentions.get(m.endIndex).add(m);
      }
      for (int j = 0 ; j < tokens.length; j++){
        if(endMentions.containsKey(j)) {
          for(Mention m : endMentions.get(j)){
            int corefChainId =  (gold)? m.goldCorefClusterID: m.corefClusterID;
            doc.append("]_").append(corefChainId);
          }
        }
        for (int k = 0 ; k < startCounts.getCount(j) ; k++) {
          char lastChar = (doc.length() > 0)? doc.charAt(doc.length()-1):' ';
          if (lastChar != '[') doc.append(" ");
          doc.append("[");
        }
        doc.append(" ");
        doc.append(tokens[j]);
      }
      if(endMentions.containsKey(tokens.length)) {
        for(Mention m : endMentions.get(tokens.length)){
          int corefChainId =  (gold)? m.goldCorefClusterID: m.corefClusterID;
          doc.append("]_").append(corefChainId); //append("_").append(m.mentionID);
        }
      }

      doc.append("\n");
    }
    logger.fine(doc.toString());
  }
  private static void printScoreSummary(String summary, Logger logger, boolean afterPostProcessing) {
    String[] lines = summary.split("\n");
    if(!afterPostProcessing) {
      boolean repe = false;
      for(String line : lines) {
        line = line.replace("Recall", "R");
        line = line.replace("Precision", "P");
        line = line.replace("oreference", "oref");   // Coreference or coreference -> Coref or coref
        if(line.startsWith("Repe")) {
          repe = true;
          logger.info(line);
        }
        if(line.startsWith("Identification of Mentions")) logger.info(line);  // mention detection score
      }
      if(repe) throw new RuntimeException();
    } else {
      StringBuilder sb = new StringBuilder();
      boolean mentionDetectionScorePrinted = false;
      int printedScoreCount = 0;
      for(String line : lines) {
        if(line.startsWith("METRIC")) sb.append(line);
        if(!mentionDetectionScorePrinted && line.startsWith("Identification of Mentions")) {
          sb.append(line).append("\n");
          mentionDetectionScorePrinted = true;
        }
        if(!line.startsWith("Identification of Mentions") && line.contains("Recall")) {
          String metric = null;
          switch(printedScoreCount++) {
            case 0:
              metric = "MUC: "; break;
            case 1:
              metric = "B^3: "; break;
            default:
              metric = "";
          }
          line = line.replace("Recall", "R");
          line = line.replace("Precision", "P");
          line = line.replace("oreference", "oref");   // Coreference or coreference -> Coref or coref
          sb.append(metric).append(line).append("\n");
        }
      }
      logger.info(sb.toString());
    }
  }

  private void printSummary(CoNLLOutputWriter conllWriter) throws IOException {
    String summary;

    // before coref (to see mention detection score)
    summary = SieveCoreferenceSystem.getConllEvalSummary(conllMentionEvalScript, conllWriter.entityGoldMentionDetectionFile, conllWriter.entityPredictedMentionDetectionFile, "muc");
    logger.info("\nENTITY (Before COREF)");
    printScoreSummary(summary, logger, false);

    summary = SieveCoreferenceSystem.getConllEvalSummary(conllMentionEvalScript, conllWriter.eventGoldMentionDetectionFile, conllWriter.eventPredictedMentionDetectionFile, "muc");
    logger.info("\nEVENT (Before COREF)");
    printScoreSummary(summary, logger, false);

    summary = SieveCoreferenceSystem.getConllEvalSummary(conllMentionEvalScript, conllWriter.bothGoldMentionDetectionFile, conllWriter.bothPredictedMentionDetectionFile, "muc");
    logger.info("\nBOTH (Before COREF)");
    printScoreSummary(summary, logger, false);

    logger.info("----------");

    // TODO ceaf is too slow for the entire corpus. need to separate it to multiple parts
    // after coref
    StringBuilder sb = new StringBuilder();
    sb.append(SieveCoreferenceSystem.getConllEvalSummary(conllMentionEvalScript, conllWriter.entityGoldCorefFile, conllWriter.entityPredictedCorefFile, "muc")).append("\n");
    sb.append(SieveCoreferenceSystem.getConllEvalSummary(conllMentionEvalScript, conllWriter.entityGoldCorefFile, conllWriter.entityPredictedCorefFile, "bcub")).append("\n");
    sb.append(SieveCoreferenceSystem.getConllEvalSummary(conllMentionEvalScript, conllWriter.entityGoldCorefFile, conllWriter.entityPredictedCorefFile, "blanc")).append("\n");
    summary = sb.toString();
    logger.info("\nENTITY (After COREF)");
    printScoreSummary(summary, logger, true);
    //    SieveCoreferenceSystem.printFinalScore(summary, logger);

    sb = new StringBuilder();
    sb.append(SieveCoreferenceSystem.getConllEvalSummary(conllMentionEvalScript, conllWriter.eventGoldCorefFile, conllWriter.eventPredictedCorefFile, "muc")).append("\n");
    sb.append(SieveCoreferenceSystem.getConllEvalSummary(conllMentionEvalScript, conllWriter.eventGoldCorefFile, conllWriter.eventPredictedCorefFile, "bcub")).append("\n");
    sb.append(SieveCoreferenceSystem.getConllEvalSummary(conllMentionEvalScript, conllWriter.eventGoldCorefFile, conllWriter.eventPredictedCorefFile, "blanc")).append("\n");
    summary = sb.toString();
    logger.info("EVENT (After COREF)");
    printScoreSummary(summary, logger, true);
    //    SieveCoreferenceSystem.printFinalScore(summary, logger);

    sb = new StringBuilder();
    sb.append(SieveCoreferenceSystem.getConllEvalSummary(conllMentionEvalScript, conllWriter.bothGoldCorefFile, conllWriter.bothPredictedCorefFile, "muc")).append("\n");
    sb.append(SieveCoreferenceSystem.getConllEvalSummary(conllMentionEvalScript, conllWriter.bothGoldCorefFile, conllWriter.bothPredictedCorefFile, "bcub")).append("\n");
    sb.append(SieveCoreferenceSystem.getConllEvalSummary(conllMentionEvalScript, conllWriter.bothGoldCorefFile, conllWriter.bothPredictedCorefFile, "blanc")).append("\n");
    summary = sb.toString();
    logger.info("BOTH (After COREF)");
    printScoreSummary(summary, logger, true);
    //    SieveCoreferenceSystem.printFinalScore(summary, logger);
  }

  // for debug
  @SuppressWarnings("unused")
  private static void printDocInfo(JointCorefDocument jDoc) {
    StringBuilder sb = new StringBuilder();

    List<List<Mention>> orderedMentions = jDoc.getOrderedMentions();

    for(List<Mention> sentMentions : orderedMentions) {
      if(sentMentions.size()==0) continue;
      // print sentence
      Mention m1 = sentMentions.get(0);
      sb.append("\n======================================\n");
      sb.append("SENTENCE: ").append(Mention.sentenceWordsToString(m1));
      sb.append("\n").append(m1.dependency);
      sb.append("\n").append(m1.contextParseTree.pennString());
      sb.append("\n\n");

      for(Mention m : sentMentions) {
        sb.append(Mention.sentenceWordsToString(m));
        sb.append("\n");
        sb.append(Mention.mentionInfo(m));
      }
      sb.append("================================================\n\n");
    }
    logger.fine(sb.toString());
  }
}
