//
// StanfordCoreNLP -- a suite of NLP tools
// Copyright (c) 2009-2011 The Board of Trustees of
// The Leland Stanford Junior University. All Rights Reserved.
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License
// as published by the Free Software Foundation; either version 2
// of the License, or (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.
//
// For more information, bug reports, fixes, contact:
//    Christopher Manning
//    Dept of Computer Science, Gates 1A
//    Stanford CA 94305-9010
//    USA
//

package edu.stanford.nlp.jcoref.dcoref;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.Set;
import java.util.logging.FileHandler;
import java.util.logging.Formatter;
import java.util.logging.Level;
import java.util.logging.LogRecord;
import java.util.logging.Logger;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import edu.stanford.nlp.classify.RVFDataset;
import edu.stanford.nlp.io.StringOutputStream;
import edu.stanford.nlp.jcoref.JointCorefClassifier;
import edu.stanford.nlp.jcoref.JCBMentionExtractor;
import edu.stanford.nlp.jcoref.RuleBasedJointCorefSystem;
import edu.stanford.nlp.jcoref.dcoref.CorefChain.CorefMention;
import edu.stanford.nlp.jcoref.dcoref.CorefChain.MentionComparator;
import edu.stanford.nlp.jcoref.dcoref.ScorerBCubed.BCubedType;
import edu.stanford.nlp.jcoref.dcoref.sievepasses.DeterministicCorefSieve;
import edu.stanford.nlp.jcoref.dcoref.sievepasses.ExactStringMatch;
import edu.stanford.nlp.jcoref.dcoref.sievepasses.JointArgumentMatch;
import edu.stanford.nlp.ling.CoreAnnotations.CharacterOffsetBeginAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.CharacterOffsetEndAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.DocIDAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.SpeakerAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TextAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.UtteranceAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.WordSenseAnnotation;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.RVFDatum;
import edu.stanford.nlp.parser.lexparser.LexicalizedParser;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.DefaultPaths;
import edu.stanford.nlp.stats.ClassicCounter;
import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.stats.Counters;
//import edu.stanford.nlp.stats.OpenAddressCounter;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.IntPair;
import edu.stanford.nlp.util.IntTuple;
import edu.stanford.nlp.util.Pair;
import edu.stanford.nlp.util.StringUtils;
import edu.stanford.nlp.util.SystemUtils;

/**
 * Multi-pass Sieve coreference resolution system (see EMNLP 2010 paper).
 *
 * @author Jenny Finkel
 * @author Mihai Surdeanu
 * @author Karthik Raghunathan
 * @author Heeyoung Lee
 * @author Sudarshan Rangarajan
 */
public class SieveCoreferenceSystem {

  public static final Logger logger = Logger.getLogger(RuleBasedJointCorefSystem.class.getName());

  /**
   * If true, we score the output of the given test document
   * Assumes gold annotations are available
   */
  private final boolean doScore;

  /**
   * If true, we do post processing.
   */
  private final boolean doPostProcessing;

  /**
   * maximum sentence distance between two mentions for resolution (-1: no constraint on distance)
   */
  private final int maxSentDist;

  /**
   * automatically set by looking at sieves
   */
  private final boolean useSemantics;

  /** flag for replicating conllst result */
  private final boolean replicateCoNLL;

  /** Path for the official CoNLL scorer  */
  private final String conllMentionEvalScript;

  /**
   * Dictionaries of all the useful goodies (gender, animacy, number etc. lists)
   */
  public final Dictionaries dictionaries;

  /**
   * Semantic knowledge: WordNet
   */
  public final Semantics semantics;

  /** Current sieve index */
  public int currentSieveIdx;

  private final String [] sieveNames;

  /** counter for links in passes (Pair<correct links, total links>)  */
  public List<Pair<Integer, Integer>> eventLinksCountInPass;
  public List<Pair<Integer, Integer>> entityLinksCountInPass;
  
  /** probabilities for singletons (Marta, Marie) */
  public static HashMap<Pair<String, Integer>, Double> singletonProbs;


  /** Scores for each pass */
  public List<CorefScorer> scorePairwiseEvent;
  public List<CorefScorer> scorePairwiseEntity;
  public List<CorefScorer> scoreBcubed;
  public List<CorefScorer> scoreMUC;

  private List<CorefScorer> eventScoreSingleDoc;
  private List<CorefScorer> entityScoreSingleDoc;

  /** Additional scoring stats */
  int additionalCorrectEventLinksCount;
  int additionalEventLinksCount;
  int additionalCorrectEntityLinksCount;
  int additionalEntityLinksCount;
  
  public JointCorefClassifier jcc;
  public static boolean trainJointCorefClassifier = true;
//  public static final String jccModelPath = "/u/heeyoung/corpus-jcb/jccModel/";
//  public static final String jccModel = jccModelPath+"/jccModel.trained.ser";
//  public static final String jccJCBModel = jccModelPath+"/jccModel.jcbdev.ser";
//  public static final String jccCoNLLModel = jccModelPath+"/jccModel.conlltrain_part.fulldcoref_joint.additionaltrainingdata.ser";

  public static class SieveStateParameter {
    public double value = Double.NaN;
    public double max;
    public double min;
    public double diff;
    public SieveStateParameter(double value, double max, double min, double diff) {
      this.value = value;
      this.max = max;
      this.min = min;
      this.diff = diff;
    }
    @Override
    public String toString() {
      StringBuilder str = new StringBuilder();
      str.append("value: ").append(value).append(", max: ").append(max).append(", min: ").append(min).append(", diff: ").append(diff);
      return str.toString();
    }
  }

  public class SieveState {
    int sievePointer = -1;
    boolean insideIteration = false;
    boolean changedInIteration = false;
    int iterationBeginPointer = -1;

    Map<String, SieveStateParameter> parameters = new HashMap<String, SieveStateParameter>();

    public boolean breakIteration() {
      boolean ret = true;
      if(changedInIteration) return false;
      for(SieveStateParameter parameter : parameters.values()) {
        if(parameter.value > parameter.min) ret = false;
      }
      for(SieveStateParameter parameter : parameters.values()) {
        parameter.value -= parameter.diff;
        if(parameter.value < parameter.min) parameter.value = parameter.min;
      }
      return ret;
    }

    public void updateChange(boolean changed) {
      changedInIteration = (changedInIteration || changed); // TODO
    }
  }
  /** Semantic knowledge: currently WordNet is available */
  public class Semantics {
    public WordNet wordnet;

    public Semantics(Dictionaries dict) throws Exception{
      wordnet = new WordNet();
    }
  }

  public static class LogFormatter extends Formatter {
    @Override
    public String format(LogRecord rec) {
      StringBuilder buf = new StringBuilder(1000);
      buf.append(formatMessage(rec));
      buf.append('\n');
      return buf.toString();
    }
  }

  public SieveCoreferenceSystem(Properties props) throws Exception {
    // initialize required fields
    currentSieveIdx = -1;

    eventLinksCountInPass = new ArrayList<Pair<Integer, Integer>>();
    entityLinksCountInPass = new ArrayList<Pair<Integer, Integer>>();
    scorePairwiseEvent = new ArrayList<CorefScorer>();
    scorePairwiseEntity = new ArrayList<CorefScorer>();
    scoreBcubed = new ArrayList<CorefScorer>();
    scoreMUC = new ArrayList<CorefScorer>();
    
    

    //
    // construct the sieve passes
    //
    String sievePasses = props.getProperty(Constants.SIEVES_PROP, Constants.SIEVEPASSES);
    sieveNames = sievePasses.trim().split(", \\s*");

    //
    // create scoring framework
    //
    doScore = Boolean.parseBoolean(props.getProperty(Constants.SCORE_PROP, "false"));

    //
    // setting post processing
    //
    doPostProcessing = Boolean.parseBoolean(props.getProperty(Constants.POSTPROCESSING_PROP, "false"));

    //
    // setting maximum sentence distance between two mentions for resolution (-1: no constraint on distance)
    //
    maxSentDist = Integer.parseInt(props.getProperty(Constants.MAXDIST_PROP, "-1"));

    //
    // set useWordNet
    //
    useSemantics = false;

    // flag for replicating conllst result
    replicateCoNLL = Boolean.parseBoolean(props.getProperty(Constants.REPLICATECONLL_PROP, "false"));
    conllMentionEvalScript = props.getProperty(Constants.CONLL_SCORER, Constants.conllMentionEvalScript);

    //    if(doScore){
    //      for(int i = 0 ; i < sieveClassNames.length ; i++){
    //        scorePairwise.add(new ScorerPairwise());
    //        scoreBcubed.add(new ScorerBCubed(BCubedType.Bconll));
    //        scoreMUC.add(new ScorerMUC());
    //        linksCountInPass.add(new Pair<Integer, Integer>(0, 0));
    //      }
    //    }

    //
    // load all dictionaries
    //
    dictionaries = new Dictionaries(props);
    semantics = (useSemantics)? new Semantics(dictionaries) : null;
    
    trainJointCorefClassifier = Boolean.parseBoolean(props.getProperty("jcoref.trainClassifier", "false"));

    if(props.getProperty("jcoref.jcbSerializedClassifier")==null
        && props.getProperty("jcoref.conllSerializedClassifier")==null) {
      jcc = new JointCorefClassifier();
    } else {
      String sers = props.getProperty("jcoref.readserializedPath");
      if(sers.contains("conll")) {
        jcc = JointCorefClassifier.load(props.getProperty("jcoref.conllSerializedClassifier"));
        System.err.println("load jccModel: "+props.getProperty("jcoref.conllSerializedClassifier"));
      }
      else {
        jcc = JointCorefClassifier.load(props.getProperty("jcoref.jcbSerializedClassifier"));
        jcc.trainData = new RVFDataset<Double, String>();
        System.err.println("load jccModel: "+props.getProperty("jcoref.jcbSerializedClassifier"));
      }
    }
  }

  private static HashMap<Pair<String, Integer>, Double> loadSingletonProbabilities(String file) {

    HashMap<Pair<String, Integer>, Double> dict = new HashMap<Pair<String, Integer>, Double>();

    try {
      BufferedReader reader = new BufferedReader(new FileReader(file));

      while(reader.ready()) {       
        String[] line = reader.readLine().toLowerCase().split(",");
        dict.put(new Pair<String, Integer>(line[1], Integer.parseInt(line[2])), Double.parseDouble(line[line.length-1]));
      } 
      reader.close();
    } catch (IOException e) {
      throw new RuntimeException(e);
    } 
    return dict;
  }

  public boolean doScore() { return doScore; }
  public Dictionaries dictionaries() { return dictionaries; }
  public Semantics semantics() { return semantics; }

  public static LexicalizedParser makeParser(Properties props) {
    int maxLen = Integer.parseInt(props.getProperty(Constants.PARSER_MAXLEN_PROP, "100"));
    String[] options = {"-maxLength", Integer.toString(maxLen)};
    LexicalizedParser parser = LexicalizedParser.loadModel(props.getProperty(Constants.PARSER_MODEL_PROP, DefaultPaths.DEFAULT_PARSER_MODEL), options);
    return parser;
  }

  /**
   * Needs the following properties:
   *  -props 'Location of coref.properties'
   * @throws Exception
   */
  public static void main(String[] args) throws Exception {
    Properties props = StringUtils.argsToProperties(args);
    String timeStamp = Calendar.getInstance().getTime().toString().replaceAll("\\s", "-");

    singletonProbs = loadSingletonProbabilities(props.getProperty("singletons.probs"));
    
    //
    // initialize logger
    //
    FileHandler fh;
    try {
      String logFileName = props.getProperty(Constants.LOG_PROP, "log.txt");
      if(logFileName.endsWith(".txt")) {
        logFileName = logFileName.substring(0, logFileName.length()-4) +"_"+ timeStamp+".txt";
      } else {
        logFileName = logFileName + "_"+ timeStamp+".txt";
      }
      fh = new FileHandler(logFileName, false);
      logger.addHandler(fh);
      logger.setLevel(Level.FINE);
      fh.setFormatter(new LogFormatter());
    } catch (SecurityException e) {
      System.err.println("ERROR: cannot initialize logger!");
      throw e;
    } catch (IOException e) {
      System.err.println("ERROR: cannot initialize logger!");
      throw e;
    }

    logger.fine(timeStamp);
    logger.fine(props.toString());
    Constants.printConstants(logger);

    // initialize coref system
    SieveCoreferenceSystem corefSystem = new SieveCoreferenceSystem(props);
    LexicalizedParser parser = makeParser(props); // Load the Stanford Parser

    // prepare conll output
    PrintWriter writerGold = null;
    PrintWriter writerPredicted = null;
    PrintWriter writerPredictedCoref = null;

    String conllOutputMentionGoldFile = null;
    String conllOutputMentionPredictedFile = null;
    String conllOutputMentionCorefPredictedFile = null;
    String conllMentionEvalFile = null;
    String conllMentionEvalErrFile = null;
    String conllMentionCorefEvalFile = null;
    String conllMentionCorefEvalErrFile = null;

    if(Constants.PRINT_CONLL_OUTPUT || corefSystem.replicateCoNLL) {
      String conllOutput = props.getProperty(Constants.CONLL_OUTPUT_PROP, "conlloutput");
      conllOutputMentionGoldFile = conllOutput + "-"+timeStamp+".gold.txt";
      conllOutputMentionPredictedFile = conllOutput +"-"+timeStamp+ ".predicted.txt";
      conllOutputMentionCorefPredictedFile = conllOutput +"-"+timeStamp+ ".coref.predicted.txt";
      conllMentionEvalFile = conllOutput +"-"+timeStamp+ ".eval.txt";
      conllMentionEvalErrFile = conllOutput +"-"+timeStamp+ ".eval.err.txt";
      conllMentionCorefEvalFile = conllOutput +"-"+timeStamp+ ".coref.eval.txt";
      conllMentionCorefEvalErrFile = conllOutput +"-"+timeStamp+ ".coref.eval.err.txt";
      logger.info("CONLL MENTION GOLD FILE: " + conllOutputMentionGoldFile);
      logger.info("CONLL MENTION PREDICTED FILE: " + conllOutputMentionPredictedFile);
      logger.info("CONLL MENTION EVAL FILE: " + conllMentionEvalFile);
      if (!Constants.SKIP_COREF) {
        logger.info("CONLL MENTION PREDICTED WITH COREF FILE: " + conllOutputMentionCorefPredictedFile);
        logger.info("CONLL MENTION WITH COREF EVAL FILE: " + conllMentionCorefEvalFile);
      }
      writerGold = new PrintWriter(new FileOutputStream(conllOutputMentionGoldFile));
      writerPredicted = new PrintWriter(new FileOutputStream(conllOutputMentionPredictedFile));
      writerPredictedCoref = new PrintWriter(new FileOutputStream(conllOutputMentionCorefPredictedFile));
    }

    // MentionExtractor extracts MUC, ACE, or CoNLL documents
    MentionExtractor mentionExtractor = null;
    if(props.containsKey(Constants.MUC_PROP)){
      mentionExtractor = new MUCMentionExtractor(parser, corefSystem.dictionaries, props, corefSystem.semantics);
    } else if(props.containsKey(Constants.ACE2004_PROP) || props.containsKey(Constants.ACE2005_PROP)) {
      mentionExtractor = new ACEMentionExtractor(parser, corefSystem.dictionaries, props, corefSystem.semantics);
    } else if (props.containsKey(Constants.CONLL2011_PROP)) {
      mentionExtractor = new CoNLLMentionExtractor(parser, corefSystem.dictionaries, props, corefSystem.semantics);
    } else if (props.containsKey(Constants.JCB_PROP)) {
      mentionExtractor = new JCBMentionExtractor(corefSystem.dictionaries, props, corefSystem.semantics);
    }
    if(mentionExtractor == null){
      throw new RuntimeException("No input file specified!");
    }
    if (!Constants.USE_GOLD_MENTIONS) {
      // Set mention finder
      String mentionFinderClass = props.getProperty(Constants.MENTION_FINDER_PROP);
      if (mentionFinderClass != null) {
        String mentionFinderPropFilename = props.getProperty(Constants.MENTION_FINDER_PROPFILE_PROP);
        CorefMentionFinder mentionFinder;
        if (mentionFinderPropFilename != null) {
          Properties mentionFinderProps = new Properties();
          mentionFinderProps.load(new FileInputStream(mentionFinderPropFilename));
          mentionFinder = (CorefMentionFinder) Class.forName(mentionFinderClass).getConstructor(Properties.class).newInstance(mentionFinderProps);
        } else {
          mentionFinder = (CorefMentionFinder) Class.forName(mentionFinderClass).newInstance();
        }
        mentionExtractor.setMentionFinder(mentionFinder);
      }
      if (mentionExtractor.mentionFinder == null) {
        logger.warning("No mention finder specified, but not using gold mentions");
      }
    }

    //
    // Parse one document at a time, and do single-doc coreference resolution in each
    //
    Document document;

    //
    // In one iteration, orderedMentionsBySentence contains a list of all
    // mentions in one document. Each mention has properties (annotations):
    // its surface form (Word), NER Tag, POS Tag, Index, etc.
    //

    while(true) {

      document = mentionExtractor.nextDoc();
      if(document==null) break;

      if(!props.containsKey(Constants.MUC_PROP)) {
        //printRawDoc(document, true);
        //printRawDoc(document, false);
      }
      //printDiscourseStructure(document);

      if(corefSystem.doScore()){
        document.extractGoldCorefClusters();
      }

      if(Constants.PRINT_CONLL_OUTPUT || corefSystem.replicateCoNLL) {
        // Not doing coref - print conll output here
        printConllOutput(document, writerGold, true);
        printConllOutput(document, writerPredicted, false);
      }

      // run mention detection only
      if(Constants.SKIP_COREF) {
        continue;
      }
      
      corefSystem.coref(document);  // Do Coreference Resolution

      if(corefSystem.doScore()){
        //Identifying possible coreferring mentions in the corpus along with any recall/precision errors with gold corpus
        corefSystem.printTopK(logger, document, corefSystem.semantics);

        //logger.fine("pairwise score for this doc: ");
        //corefSystem.scoreSingleDoc.get(corefSystem.sieves.length-1).printF1(logger);
        //logger.fine("accumulated score: ");
        //corefSystem.printF1(true);
        //logger.fine("\n");
      }
      if(Constants.PRINT_CONLL_OUTPUT || corefSystem.replicateCoNLL){
        if(Constants.REMOVE_SINGLETONS){
          printConllOutput(document, writerPredictedCoref, false, true);
        } else {
          printConllOutput(document, writerPredictedCoref, false, false);
        }
      }
    }

    if(Constants.PRINT_CONLL_OUTPUT || corefSystem.replicateCoNLL) {
      writerGold.close();
      writerPredicted.close();
      writerPredictedCoref.close();

      if(props.containsKey(Constants.CONLL_SCORER)) {
        runConllEval(corefSystem.conllMentionEvalScript, conllOutputMentionGoldFile, conllOutputMentionPredictedFile, conllMentionEvalFile, conllMentionEvalErrFile);

        String summary = getConllEvalSummary(corefSystem.conllMentionEvalScript, conllOutputMentionGoldFile, conllOutputMentionPredictedFile);
        logger.info("CONLL EVAL SUMMARY (Before COREF)\n" + summary);

        if (!Constants.SKIP_COREF) {
          runConllEval(corefSystem.conllMentionEvalScript, conllOutputMentionGoldFile, conllOutputMentionCorefPredictedFile, conllMentionCorefEvalFile, conllMentionCorefEvalErrFile);
          summary = getConllEvalSummary(corefSystem.conllMentionEvalScript, conllOutputMentionGoldFile, conllOutputMentionCorefPredictedFile);
          logger.info("CONLL EVAL SUMMARY (After COREF)\n" + summary);
          printFinalScore(summary, logger);
        }
      }
    }
    logger.info("done");
  }


  private DeterministicCorefSieve getNextSieve(SieveState state) {
    DeterministicCorefSieve nextSieve;
    Pattern p = Pattern.compile("(.*)\\((.*)\\)");
    String previousSieve = (state.sievePointer==-1)? null : sieveNames[state.sievePointer];

    // find next sieve
    if(previousSieve!=null && previousSieve.endsWith("iterationEnd")) {
      if(state.breakIteration()) {
        state.parameters = new HashMap<String, SieveStateParameter>();
        state.sievePointer++;
      }
      else {
        state.sievePointer = state.iterationBeginPointer;
        state.changedInIteration = false;
      }
    } else {
      state.sievePointer++;
    }

    // build next sieve
    try {
      String nextSieveString = (state.sievePointer < sieveNames.length)? sieveNames[state.sievePointer] : null;
      if(nextSieveString==null) return null;
      if(nextSieveString.endsWith("iterationBegin")) state.iterationBeginPointer = state.sievePointer;
      Matcher matcher = p.matcher(nextSieveString);
      if(matcher.find()) {
        String className = matcher.group(1);
        String[] options = matcher.group(2).split(",");
        StringBuilder optionForSieve = new StringBuilder();
        for(String option : options) {
          if(optionForSieve.length()!=0) optionForSieve.append(",");
          if(option.contains("-")) {
            String[] paramOption = option.split(":");
            optionForSieve.append(paramOption[0]).append(":");
            if(!state.parameters.containsKey(paramOption[0])) {
              String[] params = paramOption[1].split("-");
              SieveStateParameter stateParameter = new SieveStateParameter(Double.parseDouble(params[0]), Double.parseDouble(params[0]), Double.parseDouble(params[1]), Double.parseDouble(params[2]));
              state.parameters.put(paramOption[0], stateParameter);
            }
            optionForSieve.append(state.parameters.get(paramOption[0]).value);
          } else {
            optionForSieve.append(option);
          }
        }

        nextSieve = (DeterministicCorefSieve) Class.forName("edu.stanford.nlp.jcoref.dcoref.sievepasses."+className).getConstructor(String.class).newInstance(optionForSieve.toString());
      } else {
        nextSieve = (DeterministicCorefSieve) Class.forName("edu.stanford.nlp.jcoref.dcoref.sievepasses."+nextSieveString).getConstructor().newInstance();
      }
    } catch (Exception e) {
      throw new RuntimeException(e);
    }
    if(doScore){
      scorePairwiseEvent.add(new ScorerPairwise(true, true));
      scorePairwiseEntity.add(new ScorerPairwise(true, false));
      scoreBcubed.add(new ScorerBCubed(BCubedType.Bconll));
      scoreMUC.add(new ScorerMUC());
      eventLinksCountInPass.add(new Pair<Integer, Integer>(0, 0));
      entityLinksCountInPass.add(new Pair<Integer, Integer>(0, 0));
    }

    return nextSieve;
  }
  /**
   * Extracts coreference clusters
   * This is the main API entry point for coreference resolution
   * @throws IOException
   */
  public Map<Integer, CorefChain> coref(Document document) throws IOException {

    // Multi-pass sieve coreference resolution
    currentSieveIdx = 0;
    DeterministicCorefSieve sieve;
    boolean changed = false;
    SieveState sieveState = new SieveState();
    while((sieve = getNextSieve(sieveState))!=null) {
      logger.fine("220220 Sieve: "+sieve.flagsToString());
      // Do coreference resolution using this pass
      changed = coreference(document, sieve);
      sieveState.updateChange(changed);
      currentSieveIdx++;
    }

    // post processing (e.g., removing singletons, appositions for conll)
    if((!Constants.USE_GOLD_MENTIONS && doPostProcessing) || replicateCoNLL) postProcessing(document);

    // coref system output: CorefChain
    Map<Integer, CorefChain> result = new HashMap<Integer, CorefChain>();
    for(CorefCluster c : document.corefClusters.values()) {
      result.put(c.clusterID, new CorefChain(c, document.positions));
    }

    return result;
  }

  public static Counter<String> sameLemmaPredicates = new ClassicCounter<String>();

  // temp: for analysis
  private void checkHowManyEventCoref(Document document){
    for(Mention m1 : document.allPredictedMentions.values()){
      for(Mention m2 : document.allPredictedMentions.values()) {
        if(m1.isVerb || m2.isVerb || m1.mentionID >= m2.mentionID) continue;
        if(document.allGoldMentions.containsKey(m1.mentionID) && document.allGoldMentions.containsKey(m2.mentionID)
            && document.allGoldMentions.get(m1.mentionID).goldCorefClusterID == document.allGoldMentions.get(m2.mentionID).goldCorefClusterID
            && m1.corefClusterID!=m2.corefClusterID) {
          // entity recall error

          for(Mention p1 : m1.srlPredicates.keySet()) {
            for(Mention p2 : m2.srlPredicates.keySet()) {
              if(p1.headWord.lemma().equals(p2.headWord.lemma())
                  && m1.srlPredicates.get(p1).equals(m2.srlPredicates.get(p2))) {
                RuleBasedJointCorefSystem.logger.fine("Same Lemma Predicates->");
                RuleBasedJointCorefSystem.logger.fine("\t["+m1.spanToString() + "] in sent:"+m1.sentNum+" -> "+Mention.sentenceWordsToString(m1));
                RuleBasedJointCorefSystem.logger.fine("\t["+m2.spanToString() + "] in sent:"+m2.sentNum+" -> "+Mention.sentenceWordsToString(m2));
                RuleBasedJointCorefSystem.logger.fine("\t\t"+m1.srlPredicates);
                RuleBasedJointCorefSystem.logger.fine("\t\t"+m2.srlPredicates);
                RuleBasedJointCorefSystem.logger.fine("\t\t\t=> coreferent event? ");
                SieveCoreferenceSystem.sameLemmaPredicates.incrementCount(p1.headWord.lemma());
              }
            }
          }
        }
      }
    }
  }

  /**
   * Do coreference resolution using one sieve pass
   * @param document - an extracted document
   * @throws IOException
   */
  private boolean coreference(Document document, DeterministicCorefSieve sieve) throws IOException {

    boolean changed = false;

    if(sieve.flags.JOINT_ARG_MATCH) {
      // temp: for analysis
//      checkHowManyEventCoref(document);

      changed =  buildDocForTracing(document, sieve, dictionaries);
      
      // TODO : temp 
//      document.initializeArgMatchCounter(jcc, dictionaries, false);
//      ((JointArgumentMatch) sieve).jointArgCoref(document, dictionaries, this.jcc);
//      RuleBasedJointCorefSystem.logger.fine("docForTracing has built!!!!");
//      followTrace(document);
      //      document.initializeArgMatchCounter();
      //      changed = JointArgumentMatch.jointArgCoref(document, sieve);
      return changed;
    }

    List<List<Mention>> orderedMentionsBySentence = document.getOrderedMentions();
    Map<Integer, CorefCluster> corefClusters = document.corefClusters;
    Set<Mention> roleSet = document.roleSet;

    logger.finest("ROLE SET (Skip exact string match): ------------------");
    for(Mention m : roleSet){
      logger.finest("\t"+m.spanToString());
    }
    logger.finest("-------------------------------------------------------");

    additionalCorrectEventLinksCount = 0;
    additionalEventLinksCount = 0;
    additionalCorrectEntityLinksCount = 0;
    additionalEntityLinksCount = 0;

    for (int sentI = 0; sentI < orderedMentionsBySentence.size(); sentI++) {
      List<Mention> orderedMentions = orderedMentionsBySentence.get(sentI);

      for (int mentionI = 0; mentionI < orderedMentions.size(); mentionI++) {

        Mention m1 = orderedMentions.get(mentionI);

        // check for skip: first mention only, discourse salience
        if(sieve.skipThisMention(document, m1, corefClusters.get(m1.corefClusterID), dictionaries)) {
          continue;
        }
        
        // Skip singletons (Marta, Marie)
        /*if(m1.isSingleton){ 
          if(document.allGoldMentions.containsKey(m1.mentionID)){
            System.out.println("FALSE singleton: " + m1);
            System.out.println("SENT: " + Mention.sentenceWordsToString(m1));
            System.out.println();
          } else {
            System.out.println("TRUE singleton: " + m1);
            System.out.println("SENT: " + Mention.sentenceWordsToString(m1));
            System.out.println();
          }
          continue;
        }*/

        LOOP:
          for (int sentJ = sentI; sentJ >= 0; sentJ--) {
            List<Mention> l = sieve.getOrderedAntecedents(sentJ, sentI, orderedMentions, orderedMentionsBySentence, m1, mentionI, corefClusters, dictionaries);
            if(maxSentDist != -1 && sentI - sentJ > maxSentDist) continue;

            // Sort mentions by length whenever we have two mentions beginning at the same position and having the same head
            for(int i = 0; i < l.size(); i++) {
              for(int j = 0; j < l.size(); j++) {
                if(l.get(i).headString.equals(l.get(j).headString) &&
                    l.get(i).startIndex == l.get(j).startIndex &&
                    l.get(i).sameSentence(l.get(j)) && j > i &&
                    l.get(i).spanToString().length() > l.get(j).spanToString().length()) {
                  logger.finest("FLIPPED: "+l.get(i).spanToString()+"("+i+"), "+l.get(j).spanToString()+"("+j+")");
                  l.set(j, l.set(i, l.get(j)));
                }
              }
            }

            for (Mention m2 : l) {
              // m2 - antecedent of m1
              
              // Skip singletons (Marta, Marie)
              //if(m2.isSingleton) continue;

              if (m1.corefClusterID == m2.corefClusterID) continue;
              CorefCluster c1 = corefClusters.get(m1.corefClusterID);
              CorefCluster c2 = corefClusters.get(m2.corefClusterID);

              if (sieve.useRoleSkip()) {
                if (m1.isRoleAppositive(m2, dictionaries)) {
                  roleSet.add(m1);
                } else if (m2.isRoleAppositive(m1, dictionaries)) {
                  roleSet.add(m2);
                }
                continue;
              }

              // calculate F1 score, feature, and add training data for the classifier
              if(SieveCoreferenceSystem.trainJointCorefClassifier
                  && (JointArgumentMatch.DOPRONOUN || (!c1.representative.isPronominal() && !c2.representative.isPronominal()))
//                  && jcc.regressor==null) {
                  ) {
                JointArgumentMatch.calculateCentroid(c1, document, true);
                JointArgumentMatch.calculateCentroid(c2, document, true);
                double s = JointArgumentMatch.getMergeScore(document, c1, c2);
                if(Math.abs(s-0.5) > 0.0001){
                  jcc.addData(new RVFDatum<Double, String>(JointArgumentMatch.getFeatures(document, c1, c2, true, dictionaries), s));
                }
              }

              if (sieve.coreferent(document, c1, c2, m1, m2, dictionaries, roleSet, semantics)) {
                if(m1.isSingleton){
                  if(!document.allGoldMentions.containsKey(m1.mentionID)){
                    System.out.println("P catch\tm1\t"+ m1 + "\t" + m2 + "\t" + Mention.sentenceWordsToString(m1));
                  } else if(document.allGoldMentions.containsKey(m1.mentionID) && document.allGoldMentions.containsKey(m2.mentionID)
                      && document.allGoldMentions.get(m1.mentionID).goldCorefClusterID == document.allGoldMentions.get(m2.mentionID).goldCorefClusterID) {
                    System.out.println("R loss\tm1\t"+ m1 + "\t" + m2 + "\t" + Mention.sentenceWordsToString(m1));
                  }
                }
                if(m2.isSingleton){
                  if(!document.allGoldMentions.containsKey(m2.mentionID)){
                    System.out.println("P catch\tm2\t"+ m1 + "\t" + m2 + "\t" + Mention.sentenceWordsToString(m2));
                  } else if(document.allGoldMentions.containsKey(m1.mentionID) && document.allGoldMentions.containsKey(m2.mentionID)
                      && document.allGoldMentions.get(m1.mentionID).goldCorefClusterID == document.allGoldMentions.get(m2.mentionID).goldCorefClusterID) {
                    System.out.println("R loss\tm2\t"+ m1 + "\t" + m2 + "\t" + Mention.sentenceWordsToString(m2));
                  }
                }

                changed = true;

                // print logs for analysis
                if (doScore()) {
                  printLogs(c1, c2, m1, m2, document, currentSieveIdx);
                }

                int removeID = c1.clusterID;
                CorefCluster.mergeClusters(c2, c1);
                corefClusters.remove(removeID);
                break LOOP;
              }
            }
          } // End of "LOOP"
      }
    }

    // scoring
    //    if(doScore()){
    //      scoreMUC.get(currentSieveIdx).calculateScore(document);
    //      scoreBcubed.get(currentSieveIdx).calculateScore(document);
    //      scorePairwiseEvent.get(currentSieveIdx).calculateScore(document);
    //      scorePairwiseEntity.get(currentSieveIdx).calculateScore(document);
    //      if(currentSieveIdx==0) {
    //        eventScoreSingleDoc = new ArrayList<CorefScorer>();
    //        entityScoreSingleDoc = new ArrayList<CorefScorer>();
    //        eventScoreSingleDoc.add(new ScorerPairwise(true, true));
    //        entityScoreSingleDoc.add(new ScorerPairwise(true, false));
    //        eventScoreSingleDoc.get(currentSieveIdx).calculateScore(document);
    //        entityScoreSingleDoc.get(currentSieveIdx).calculateScore(document);
    //
    //        additionalCorrectEventLinksCount = (int) eventScoreSingleDoc.get(currentSieveIdx).precisionNumSum;
    //        additionalEventLinksCount = (int) eventScoreSingleDoc.get(currentSieveIdx).precisionDenSum;
    //
    //        additionalCorrectEntityLinksCount = (int) entityScoreSingleDoc.get(currentSieveIdx).precisionNumSum;
    //        additionalEntityLinksCount = (int) entityScoreSingleDoc.get(currentSieveIdx).precisionDenSum;
    //      } else {
    //        eventScoreSingleDoc.add(new ScorerPairwise(true, true));
    //        eventScoreSingleDoc.get(currentSieveIdx).calculateScore(document);
    //        additionalCorrectEventLinksCount = (int) (eventScoreSingleDoc.get(currentSieveIdx).precisionNumSum - eventScoreSingleDoc.get(currentSieveIdx-1).precisionNumSum);
    //        additionalEventLinksCount = (int) (eventScoreSingleDoc.get(currentSieveIdx).precisionDenSum - eventScoreSingleDoc.get(currentSieveIdx-1).precisionDenSum);
    //
    //        entityScoreSingleDoc.add(new ScorerPairwise(true, false));
    //        entityScoreSingleDoc.get(currentSieveIdx).calculateScore(document);
    //        additionalCorrectEntityLinksCount = (int) (entityScoreSingleDoc.get(currentSieveIdx).precisionNumSum - entityScoreSingleDoc.get(currentSieveIdx-1).precisionNumSum);
    //        additionalEntityLinksCount = (int) (entityScoreSingleDoc.get(currentSieveIdx).precisionDenSum - entityScoreSingleDoc.get(currentSieveIdx-1).precisionDenSum);
    //      }
    //      eventLinksCountInPass.get(currentSieveIdx).setFirst(eventLinksCountInPass.get(currentSieveIdx).first() + additionalCorrectEventLinksCount);
    //      eventLinksCountInPass.get(currentSieveIdx).setSecond(eventLinksCountInPass.get(currentSieveIdx).second() + additionalEventLinksCount);
    //
    //      entityLinksCountInPass.get(currentSieveIdx).setFirst(entityLinksCountInPass.get(currentSieveIdx).first() + additionalCorrectEntityLinksCount);
    //      entityLinksCountInPass.get(currentSieveIdx).setSecond(entityLinksCountInPass.get(currentSieveIdx).second() + additionalEntityLinksCount);
    //
    //      printSieveScore(document, sieve);
    //      for(int id : corefClusters.keySet()){
    //        CorefCluster c = corefClusters.get(id);
    //        c.printCorefCluster(logger, document);
    //        logger.fine("");
    //      }
    //    }
    return changed;
  }

  private void followTrace(Document document) {
    if(document.docForTracing.calinskiScore.size()==0) return;
    int optimalClusterSize = Counters.argmax(document.docForTracing.calinskiScore);
    RuleBasedJointCorefSystem.logger.fine("optimalClusterSize: "+optimalClusterSize+", calinski score: "+document.docForTracing.calinskiScore.getCount(optimalClusterSize));
    for(IntPair idPair : document.docForTracing.mergingList) {
      if(document.corefClusters.size() == optimalClusterSize) break;
      int mergedID = idPair.get(0);
      int removeID = idPair.get(1);
      CorefCluster cMerged = document.corefClusters.get(mergedID);
      CorefCluster cRemove = document.corefClusters.get(removeID);
      //      JointArgumentMatch.logClusterMerge(document, cMerged, cRemove, idPair, true);
      CorefCluster.mergeClusters(cMerged, cRemove);
      document.corefClusters.remove(removeID);
    }
  }

  private boolean buildDocForTracing(Document document, DeterministicCorefSieve sieve, Dictionaries dict) throws IOException {
    //    JointArgumentMatch.startTime = new Date();
//    Document.copyClusterInfo(document, document.docForTracing);
//    if(SieveCoreferenceSystem.trainJointCorefClassifier) {
//      document.docForTracing.initializeArgMatchCounter(jcc, dict);
//      ((JointArgumentMatch) sieve).jointArgCoref(document.docForTracing, dict, this.jcc);
//    } else {
    JointArgumentMatch.rawFeatures = new HashMap<IntPair, RVFDatum<Double, String>> ();
    document.initializeArgMatchCounter(jcc, dict, true);
    return ((JointArgumentMatch) sieve).jointArgCoref(document, dict, this.jcc);
//    }
  }

  /** Remove singletons, appositive, predicate nominatives, relative pronouns */
  private void postProcessing(Document document) {
    Set<IntTuple> removeSet = new HashSet<IntTuple>();
    Set<Integer> removeClusterSet = new HashSet<Integer>();

    for(CorefCluster c : document.corefClusters.values()){
      Set<Mention> removeMentions = new HashSet<Mention>();
      for(Mention m : c.getCorefMentions()) {
        if(Constants.REMOVE_APPOSITION_PREDICATENOMINATIVES
            && ((m.appositions!=null && m.appositions.size() > 0)
                || (m.predicateNominatives!=null && m.predicateNominatives.size() > 0)
                || (m.relativePronouns!=null && m.relativePronouns.size() > 0))){
          removeMentions.add(m);
          removeSet.add(document.positions.get(m));
          m.corefClusterID = m.mentionID;
        }
      }
      c.corefMentions.removeAll(removeMentions);
      if(Constants.REMOVE_SINGLETONS && c.getCorefMentions().size()==1) {
        removeClusterSet.add(c.clusterID);
      }
    }
    for(int removeId : removeClusterSet){
      document.corefClusters.remove(removeId);
    }
    for(IntTuple pos : removeSet){
      document.positions.remove(pos);
    }
  }
  /** Remove singleton clusters */
  public static List<List<Mention>> filterMentionsWithSingletonClusters(
      Document document, List<List<Mention>> mentions) {

    List<List<Mention>> res = new ArrayList<List<Mention>>(mentions.size());
    for (List<Mention> ml:mentions) {
      List<Mention> filtered = new ArrayList<Mention>();
      for (Mention m:ml) {
        CorefCluster cluster = document.corefClusters.get(m.corefClusterID);
        if (cluster != null && cluster.getCorefMentions().size() > 1) {
          filtered.add(m);
        }
      }
      res.add(filtered);
    }
    return res;
  }
  public static void runConllEval(String conllMentionEvalScript,
      String goldFile, String predictFile, String evalFile, String errFile) throws IOException
      {
    ProcessBuilder process = new ProcessBuilder(conllMentionEvalScript, "all", goldFile, predictFile);
    PrintWriter out = new PrintWriter(new FileOutputStream(evalFile));
    PrintWriter err = new PrintWriter(new FileOutputStream(errFile));
    SystemUtils.run(process, out, err);
    out.close();
    err.close();
      }

  public static String getConllEvalSummary(String conllMentionEvalScript,
      String goldFile, String predictFile) throws IOException {
    return getConllEvalSummary(conllMentionEvalScript, goldFile, predictFile, "all");
  }

  public static String getConllEvalSummary(String conllMentionEvalScript,
      String goldFile, String predictFile, String metric) throws IOException {
    ProcessBuilder process = new ProcessBuilder(/*"C:\\strawberry\\perl\\bin\\perl.exe",*/conllMentionEvalScript, metric, goldFile, predictFile, "none");
    StringOutputStream errSos = new StringOutputStream();
    StringOutputStream outSos = new StringOutputStream();
    PrintWriter out = new PrintWriter(outSos);
    PrintWriter err = new PrintWriter(errSos);
    //System.out.println(/*"C:\\strawberry\\perl\\bin\\perl.exe " +*/ conllMentionEvalScript+ " "+ metric+ " "+ goldFile+ " "+ predictFile+ " none");
    //Process p = Runtime.getRuntime().exec("C:\\strawberry\\perl\\bin\\perl.exe " + conllMentionEvalScript+ " "+ metric+ " "+ goldFile+ " "+ predictFile+ " none");
    SystemUtils.run( process, out, err);
    out.close();
    err.close();
    String summary = outSos.toString();
    String errStr = errSos.toString();
    if (errStr.length() > 0) {
      summary += "\nERROR: " + errStr;
    }
    String line;
    //String summary =  null;
    
    //BufferedReader in = new BufferedReader(new InputStreamReader(p.getInputStream()) );
   // while ((line = in.readLine()) != null) {
    //	summary = summary + line;
    //}
    //in.close();
    //String summary = p.getInputStream().toString();
    		
    return summary;
  }

  /** Print logs for error analysis */
  public void printTopK(Logger logger, Document document, Semantics semantics) {

    List<List<Mention>> orderedMentionsBySentence = document.getOrderedMentions();
    Map<Integer, CorefCluster> corefClusters = document.corefClusters;
    Map<Integer, Mention> golds = document.allGoldMentions;

    logger.fine("=======ERROR ANALYSIS=========================================================");

    boolean correct;
    boolean chosen;
    for(int i = 0 ; i < orderedMentionsBySentence.size(); i++){
      for(int j =0 ; j < orderedMentionsBySentence.get(i).size(); j++){
        Mention m = orderedMentionsBySentence.get(i).get(j);
        List<Mention> orderedMentions = orderedMentionsBySentence.get(i);
        CorefCluster corefCluster = corefClusters.get(m.corefClusterID);
        if (corefCluster != null) {
          logger.fine("=========Line: "+i+"\tmention: "+j+"=======================================================");
          logger.fine(m.spanToString()+"\tmentionID: "+m.mentionID+"\tcorefClusterID: "+m.corefClusterID+"\tgoldCorefClusterID: "+m.goldCorefClusterID);
          //          corefCluster.printCorefCluster(logger);
        } else {
          continue;
        }
        logger.fine("-------------------------------------------------------");

        // to remove cascaded errors
        boolean oneRecallErrorPrinted = false;
        boolean onePrecisionErrorPrinted = false;
        boolean alreadyChoose = false;

        for (int sentJ = i; sentJ >= 0; sentJ--) {
          List<Mention> l = (new ExactStringMatch()).getOrderedAntecedents(sentJ, i, orderedMentions, orderedMentionsBySentence, m, j, corefClusters, dictionaries);

          // Sort mentions by length whenever we have two mentions beginning at the same position and having the same head
          for(int ii = 0; ii < l.size(); ii++) {
            for(int jj = 0; jj < l.size(); jj++) {
              if(l.get(ii).headString.equals(l.get(jj).headString) &&
                  l.get(ii).startIndex == l.get(jj).startIndex &&
                  l.get(ii).sameSentence(l.get(jj)) && jj > ii &&
                  l.get(ii).spanToString().length() > l.get(jj).spanToString().length()) {
                logger.finest("FLIPPED: "+l.get(ii).spanToString()+"("+ii+"), "+l.get(jj).spanToString()+"("+jj+")");
                l.set(jj, l.set(ii, l.get(jj)));
              }
            }
          }

          logger.finest("Candidates in sentence #"+sentJ+" for mention: "+m.spanToString());
          for(int ii = 0; ii < l.size(); ii ++){
            logger.finest("\tCandidate #"+ii+": "+l.get(ii).spanToString());
          }

          for (Mention antecedent : l) {
            if(corefClusters.get(antecedent.corefClusterID)==null) continue;
            chosen=(m.corefClusterID==antecedent.corefClusterID);

            // now print only errors between gold mentions
            //            if(!golds.containsKey(m.mentionID) || !golds.containsKey(antecedent.mentionID)) continue;
            //            if(!golds.containsKey(m.mentionID) || document.goldOrderedMentionsBySentence.get(antecedent.sentNum).size()==0) continue;
            if(m.sentNum < 0 
                || document.goldOrderedMentionsBySentence.get(m.sentNum).size()==0
                || document.goldOrderedMentionsBySentence.get(antecedent.sentNum).size()==0) continue;
            boolean coreferent = golds.containsKey(m.mentionID)
            && golds.containsKey(antecedent.mentionID)
            && (golds.get(m.mentionID).goldCorefClusterID == golds.get(antecedent.mentionID).goldCorefClusterID);
            correct=(chosen==coreferent);

            String chosenness = chosen ? "Chosen" : "Not Chosen";
            String correctness = correct ? "Correct" : "Incorrect";

            logger.fine("\t" + correctness +"\t\t" + chosenness + "\t"+antecedent.spanToString());
            CorefCluster mC = corefClusters.get(m.corefClusterID);
            CorefCluster aC = corefClusters.get(antecedent.corefClusterID);
            IntPair sentPair = new IntPair(Math.min(m.sentNum, antecedent.sentNum), Math.max(m.sentNum, antecedent.sentNum));

            // print precision error
            if(chosen && !correct && !onePrecisionErrorPrinted && !alreadyChoose)  {
              onePrecisionErrorPrinted = true;

              double sim = (m.sentNum==antecedent.sentNum)? 1 : document.sent2ndOrderSimilarity.get(sentPair);
              logger.fine("\nsentence dist 2nd order similarity score: "+sim);
              sim = (m.sentNum==antecedent.sentNum)? 1 : document.sent1stOrderSimilarity.get(sentPair);
              logger.fine("\nsentence dist 1st order similarity score: "+sim);

              String errorType = (m.isEvent || (document.allGoldMentions.containsKey(m.mentionID) && document.allGoldMentions.get(m.mentionID).isEvent))? "\nEVENT PRECISION ERROR " : "\nENTITY PRECISION ERROR ";
              printLinkWithContext(logger, errorType, m, antecedent, document, semantics);

              logger.fine(Mention.mentionInfo(m));
              logger.fine("--------------------------------");
              logger.fine(Mention.mentionInfo(antecedent));
              logger.fine("======================");
              logger.fine("\nEND of PRECISION ERROR LOG\n");
            }

            // print recall error
            if(!chosen && !correct && !oneRecallErrorPrinted && (!alreadyChoose || (alreadyChoose && onePrecisionErrorPrinted))) {
              oneRecallErrorPrinted = false;

              double sim = (m.sentNum==antecedent.sentNum)? 1 : document.sent2ndOrderSimilarity.get(sentPair);
              logger.fine("\nsentence dist similarity score: "+sim);

              String errorType = (m.isEvent || (document.allGoldMentions.containsKey(m.mentionID) && document.allGoldMentions.get(m.mentionID).isEvent))? "\nEVENT RECALL ERROR " : "\nENTITY RECALL ERROR ";
              printLinkWithContext(logger, errorType, m, antecedent, document, semantics);

              IntPair clusterPair = new IntPair(Math.min(mC.clusterID, aC.clusterID), Math.max(mC.clusterID, aC.clusterID));
              if(jcc.regressor!=null) {
                Counter<String> features = JointArgumentMatch.getFeatures(document, mC, aC, false, this.dictionaries);
                logger.fine("Regression Score: "+jcc.valueOf(new RVFDatum<Double, String>(features)));
                logger.fine("srlAgreeCount: "+features.getCount("SRLAGREECOUNT"));
              }

              logger.fine(Mention.mentionInfo(m));
              logger.fine("--------------------------------");
              logger.fine(Mention.mentionInfo(antecedent));
              logger.fine("======================");
              logger.fine("cluster info: ");
              if (mC != null) {
                mC.printCorefCluster(logger, document);
              } else {
                logger.finer("CANNOT find coref cluster for cluster " + m.corefClusterID);
              }
              logger.fine("----------------------------------------------------------");
              if (aC != null) {
                aC.printCorefCluster(logger, document);
              } else {
                logger.finer("CANNOT find coref cluster for cluster " + m.corefClusterID);
              }
              logger.fine("\nEND of RECALL ERROR LOG\n");
            }
            if(chosen) alreadyChoose = true;
          }
        }
        logger.fine("\n");
      }
    }
    logger.fine("===============================================================================");
  }

  public void printF1(boolean printF1First) {
    scoreMUC.get(sieveNames.length - 1).printF1(logger, printF1First);
    scoreBcubed.get(sieveNames.length - 1).printF1(logger, printF1First);
    scorePairwiseEvent.get(sieveNames.length - 1).printF1(logger, printF1First);
    scorePairwiseEntity.get(sieveNames.length - 1).printF1(logger, printF1First);
  }

  private void printSieveScore(Document document, DeterministicCorefSieve sieve) {
    logger.fine("===========================================");
    logger.fine("pass"+currentSieveIdx+": "+ sieve.flagsToString());
    scoreMUC.get(currentSieveIdx).printF1(logger);
    scoreBcubed.get(currentSieveIdx).printF1(logger);
    scorePairwiseEvent.get(currentSieveIdx).printF1(logger);
    scorePairwiseEntity.get(currentSieveIdx).printF1(logger);
    logger.fine("# of Clusters: "+document.corefClusters.size() + ",\t# of additional event links: "+additionalEventLinksCount
        +",\t# of additional correct event links: "+additionalCorrectEventLinksCount
        +",\tprecision of new event links: "+1.0*additionalCorrectEventLinksCount/additionalEventLinksCount);
    logger.fine("# of total additional event links: "+eventLinksCountInPass.get(currentSieveIdx).second()
        +",\t# of total additional correct event links: "+eventLinksCountInPass.get(currentSieveIdx).first()
        +",\taccumulated precision of this pass: "+1.0*eventLinksCountInPass.get(currentSieveIdx).first()/eventLinksCountInPass.get(currentSieveIdx).second());

    logger.fine("# of Clusters: "+document.corefClusters.size() + ",\t# of additional entity links: "+additionalEntityLinksCount
        +",\t# of additional correct entity links: "+additionalCorrectEntityLinksCount
        +",\tprecision of new entity links: "+1.0*additionalCorrectEntityLinksCount/additionalEntityLinksCount);
    logger.fine("# of total additional entity links: "+entityLinksCountInPass.get(currentSieveIdx).second()
        +",\t# of total additional correct entity links: "+entityLinksCountInPass.get(currentSieveIdx).first()
        +",\taccumulated precision of this pass: "+1.0*entityLinksCountInPass.get(currentSieveIdx).first()/entityLinksCountInPass.get(currentSieveIdx).second());

    logger.fine("--------------------------------------");
  }

  /** Print coref link info */
  private static void printLink(Logger logger, String header, Mention m, Mention ant) {
    StringBuilder sb = new StringBuilder();
    sb.append(header).append(": [").append(m.spanToString()).append("](id=").append(m.mentionID).append(") in sent #").append(m.sentNum);
    sb.append(" => [").append(ant.spanToString()).append("](id=").append(ant.mentionID).append(") in sent #").append(ant.sentNum);
    if(m.sentNum == ant.sentNum) sb.append(" Same Sentence");
    logger.fine(sb.toString());
  }

  public static void printList(Logger logger, String... args)  {
    StringBuilder sb = new StringBuilder();
    for(String arg : args)
      sb.append(arg).append("\t");
    logger.fine(sb.toString());
  }

  /** print a coref link information including context and parse tree */
  private static void printLinkWithContext(Logger logger,
      String header,
      Mention srcMention,
      Mention dstMention,
      Document document, Semantics semantics
  ) {
    List<List<Mention>> goldOrderedMentionsBySentence = document.goldOrderedMentionsBySentence;

    printLink(logger, header, srcMention, dstMention);
    IntPair idPair = new IntPair(Math.min(srcMention.mentionID, dstMention.mentionID), Math.max(srcMention.mentionID, dstMention.mentionID));
    String errorReason = "Error Reason: ";
    if(document.errorLog.containsKey(idPair)) {
      errorReason += document.errorLog.get(idPair);
    }
    logger.fine(errorReason);

    if(!srcMention.isEvent) {
      printList(logger, "Mention:" + srcMention.spanToString(),
          "Gender:" + srcMention.gender.toString(),
          "Number:" + srcMention.number.toString(),
          "Animacy:" + srcMention.animacy.toString(),
          "Person:" + srcMention.person.toString(),
          "NER:" + srcMention.nerString,
          "Head:" + srcMention.headString,
          "Type:" + srcMention.mentionType.toString(),
          "utter: "+srcMention.headWord.get(UtteranceAnnotation.class),
          "speakerID: "+srcMention.headWord.get(SpeakerAnnotation.class),
          "twinless:" + srcMention.twinless,
          "srlPredicate:" + srcMention.srlPredicates);
    } else {
      printList(logger,
          "\nisVerb:"+srcMention.isVerb,
          "isReport:"+srcMention.isReport,
          "SuperSense:"+srcMention.headWord.get(WordSenseAnnotation.class),
          "twinless:" + srcMention.twinless,
          "location:"+srcMention.location,
          "time:"+srcMention.time,
          "Roles:"+srcMention.srlArgs.toString(),
          "srlPredicate:" + srcMention.srlPredicates
      );
    }

    // shared mention in both sentences
    Set<Integer> sharedMentionClusterID = new HashSet<Integer>();
    Set<Integer> temp = new HashSet<Integer>();
    for(Mention m : goldOrderedMentionsBySentence.get(srcMention.sentNum)) {
      sharedMentionClusterID.add(m.goldCorefClusterID);
    }
    for(Mention a : goldOrderedMentionsBySentence.get(dstMention.sentNum)) {
      temp.add(a.goldCorefClusterID);
    }
    sharedMentionClusterID.retainAll(temp);

    // print context
    StringBuilder golds = new StringBuilder();
    golds.append("CONTEXT:\n");
    //Counter<Integer> mBegin = new OpenAddressCounter<Integer>();
    //Counter<Integer> mEnd = new OpenAddressCounter<Integer>();
    HashMap<Integer, Set<Mention>> endMentions = new HashMap<Integer, Set<Mention>>();

    for(Mention m : goldOrderedMentionsBySentence.get(srcMention.sentNum)){
      if(!sharedMentionClusterID.contains(m.goldCorefClusterID)) continue;
  //    mBegin.incrementCount(m.startIndex);
    //  mEnd.incrementCount(m.endIndex);
      if(!endMentions.containsKey(m.endIndex)) endMentions.put(m.endIndex, new HashSet<Mention>());
      endMentions.get(m.endIndex).add(m);
    }
    List<CoreLabel> l = document.annotation.get(SentencesAnnotation.class).get(srcMention.sentNum).get(TokensAnnotation.class);
    for(int i = 0 ; i < l.size() ; i++){
      if(endMentions.containsKey(i)) {
        for(Mention m : endMentions.get(i)) {
          golds.append("]_").append(m.goldCorefClusterID).append(" ");
        }
      }
      /*for(int j = 0; j < mBegin.getCount(i); j++){
        golds.append("[");
      }*/
      golds.append(l.get(i).get(TextAnnotation.class));
      golds.append(" ");
    }
    golds.append("\n");
    logger.fine(golds.toString());

    if(!dstMention.isEvent) {
      printList(logger, "\nAntecedent:" + dstMention.spanToString(),
          "Gender:" + dstMention.gender.toString(),
          "Number:" + dstMention.number.toString(),
          "Animacy:" + dstMention.animacy.toString(),
          "Person:" + dstMention.person.toString(),
          "NER:" + dstMention.nerString,
          "Head:" + dstMention.headString,
          "Type:" + dstMention.mentionType.toString(),
          "utter: "+dstMention.headWord.get(UtteranceAnnotation.class),
          "speakerID: "+dstMention.headWord.get(SpeakerAnnotation.class),
          "twinless:" + dstMention.twinless,
          "srlPredicate:" +dstMention.srlPredicates);
    } else {
      printList(logger,
          "\nisVerb:"+dstMention.isVerb,
          "isReport:"+dstMention.isReport,
          "SuperSense:"+dstMention.headWord.get(WordSenseAnnotation.class),
          "twinless:" + dstMention.twinless,
          "location:"+dstMention.location,
          "time:"+dstMention.time,
          "Roles:"+dstMention.srlArgs.toString(),
          "srlPredicate:" + dstMention.srlPredicates
      );
    }

    // print context
    golds = new StringBuilder();
    golds.append("CONTEXT:\n");
    //mBegin = new OpenAddressCounter<Integer>();
    //mEnd = new OpenAddressCounter<Integer>();
    endMentions = new HashMap<Integer, Set<Mention>>();

    for(Mention m : goldOrderedMentionsBySentence.get(dstMention.sentNum)){
      if(!sharedMentionClusterID.contains(m.goldCorefClusterID)) continue;
     // mBegin.incrementCount(m.startIndex);
      //mEnd.incrementCount(m.endIndex);
      if(!endMentions.containsKey(m.endIndex)) endMentions.put(m.endIndex, new HashSet<Mention>());
      endMentions.get(m.endIndex).add(m);
    }
    l = document.annotation.get(SentencesAnnotation.class).get(dstMention.sentNum).get(TokensAnnotation.class);
    for(int i = 0 ; i < l.size() ; i++){
      if(endMentions.containsKey(i)) {
        for(Mention m : endMentions.get(i)) {
          golds.append("]_").append(m.goldCorefClusterID).append(" ");
        }
      }
      /*for(int j = 0; j < mBegin.getCount(i); j++){
        golds.append("[");
      }*/
      golds.append(l.get(i).get(TextAnnotation.class));
      golds.append(" ");
    }
    golds.append("\n");
    logger.fine(golds.toString());

    // print parsed tree
    logger.finer("\nMention:: --------------------------------------------------------");
    try {
      logger.finer(srcMention.dependency.toString());
    } catch (Exception e){} //throw new RuntimeException(e);}
    logger.finer("Parse:");
    logger.finer(formatPennTree(srcMention.contextParseTree));
    logger.finer("\nAntecedent:: -----------------------------------------------------");
    try {
      logger.finer(dstMention.dependency.toString());
    } catch (Exception e){} //throw new RuntimeException(e);}
    logger.finer("Parse:");
    logger.finer(formatPennTree(dstMention.contextParseTree));
  }
  /** For printing tree in a better format */
  public static String formatPennTree(Tree parseTree)	{
    String treeString = parseTree.pennString();
    treeString = treeString.replaceAll("\\[TextAnnotation=", "");
    treeString = treeString.replaceAll("(NamedEntityTag|Value|Index|PartOfSpeech)Annotation.+?\\)", ")");
    treeString = treeString.replaceAll("\\[.+?\\]", "");
    return treeString;
  }

  /** Print pass results */
  private static void printLogs(CorefCluster c1, CorefCluster c2, Mention m1,
      Mention m2, Document document, int sieveIndex) {

    boolean correctness = false;
    Map<Integer, Mention> goldMentions = document.allGoldMentions;
    if(goldMentions.containsKey(m1.mentionID) && goldMentions.containsKey(m2.mentionID)
        && goldMentions.get(m1.mentionID).goldCorefClusterID == goldMentions.get(m2.mentionID).goldCorefClusterID) {
      correctness = true;
    }

    String correct = (correctness)? "\tCorrect" : "\tIncorrect";

    if(!m1.twinless && !m2.twinless) {
      if(!correctness){
        logger.fine("-------Incorrect merge in pass"+sieveIndex+"::--------------------");
        c1.printCorefCluster(logger, document);
        logger.fine("--------------------------------------------");
        c2.printCorefCluster(logger, document);
        logger.fine("--------------------------------------------");
      }
      logger.fine("antecedent: "+m2.spanToString()+"("+m2.mentionID+")\tmention: "+m1.spanToString()+"("+m1.mentionID+")\tsentDistance: "+Math.abs(m1.sentNum-m2.sentNum)+"\t"+correct+" Pass"+sieveIndex+":");
    }
  }

  private static void printDiscourseStructure(Document document) {
    logger.finer("DISCOURSE STRUCTURE==============================");
    logger.finer("doc type: "+document.docType);
    int previousUtterIndex = -1;
    String previousSpeaker = "";
    StringBuilder sb = new StringBuilder();
    for(CoreMap s : document.annotation.get(SentencesAnnotation.class)) {
      for(CoreLabel l : s.get(TokensAnnotation.class)) {
        int utterIndex = l.get(UtteranceAnnotation.class);
        String speaker = l.get(SpeakerAnnotation.class);
        String word = l.get(TextAnnotation.class);
        if(previousUtterIndex!=utterIndex) {
          try {
            int previousSpeakerID = Integer.parseInt(previousSpeaker);
            logger.finer("\n<utter>: "+previousUtterIndex + " <speaker>: "+document.allPredictedMentions.get(previousSpeakerID).spanToString());
          } catch (Exception e) {
            logger.finer("\n<utter>: "+previousUtterIndex + " <speaker>: "+previousSpeaker);
          }

          logger.finer(sb.toString());
          sb.setLength(0);
          previousUtterIndex = utterIndex;
          previousSpeaker = speaker;
        }
        sb.append(" ").append(word);
      }
      sb.append("\n");
    }
    try {
      int previousSpeakerID = Integer.parseInt(previousSpeaker);
      logger.finer("\n<utter>: "+previousUtterIndex + " <speaker>: "+document.allPredictedMentions.get(previousSpeakerID).spanToString());
    } catch (Exception e) {
      logger.finer("\n<utter>: "+previousUtterIndex + " <speaker>: "+previousSpeaker);
    }
    logger.finer(sb.toString());
    logger.finer("END OF DISCOURSE STRUCTURE==============================");
  }

  /** Print average F1 of MUC, B^3, CEAF_E */
  public static void printFinalScore(String summary, Logger logger) {
    Pattern f1 = Pattern.compile("Coreference:.*F1: (.*)%");
    Matcher f1Matcher = f1.matcher(summary);
    double[] F1s = new double[5];
    int i = 0;
    while (f1Matcher.find()) {
      F1s[i++] = Double.parseDouble(f1Matcher.group(1));
    }
    logger.info("Final score ((muc+bcub+ceafe)/3) = "+(F1s[0]+F1s[1]+F1s[3])/3);
  }

  public static void printConllOutput(Document document, PrintWriter writer, boolean gold) {
    printConllOutput(document, writer, gold, false);
  }

  public static void printConllOutput(Document document, PrintWriter writer, boolean gold, boolean filterSingletons) {
    List<List<Mention>> orderedMentions;
    if(gold) orderedMentions = document.goldOrderedMentionsBySentence;
    else orderedMentions = document.predictedOrderedMentionsBySentence;
    if (filterSingletons) {
      orderedMentions = filterMentionsWithSingletonClusters(document, orderedMentions);
    }
    printConllOutput(document, writer, orderedMentions, gold);
  }

  public static void printConllOutput(Document document, PrintWriter writer, List<List<Mention>> orderedMentions, boolean gold)
  {
    Annotation anno = document.annotation;
    //List<List<String[]>> conllDocSentences = document.conllDoc.sentenceWordLists;
    String docID = anno.get(DocIDAnnotation.class);
    StringBuilder sb = new StringBuilder();
    sb.append("#begin document ").append(docID).append("\n");
    List<CoreMap> sentences = anno.get(SentencesAnnotation.class);
    for(int sentNum = 0 ; sentNum < sentences.size() ; sentNum++){
      List<CoreLabel> sentence = sentences.get(sentNum).get(TokensAnnotation.class);
      //List<String[]> conllSentence = conllDocSentences.get(sentNum);
      Map<Integer,Set<Mention>> mentionBeginOnly = new HashMap<Integer,Set<Mention>>();
      Map<Integer,Set<Mention>> mentionEndOnly = new HashMap<Integer,Set<Mention>>();
      Map<Integer,Set<Mention>> mentionBeginEnd = new HashMap<Integer,Set<Mention>>();

      for(int i=0 ; i<sentence.size(); i++){
        mentionBeginOnly.put(i, new LinkedHashSet<Mention>());
        mentionEndOnly.put(i, new LinkedHashSet<Mention>());
        mentionBeginEnd.put(i, new LinkedHashSet<Mention>());
      }

      for(Mention m : orderedMentions.get(sentNum)) {
        if(m.startIndex==m.endIndex-1) {
          mentionBeginEnd.get(m.startIndex).add(m);
        } else {
          mentionBeginOnly.get(m.startIndex).add(m);
          mentionEndOnly.get(m.endIndex-1).add(m);
        }
      }

      for(int i=0 ; i<sentence.size(); i++){
        StringBuilder sb2 = new StringBuilder();
        for(Mention m : mentionBeginOnly.get(i)){
          if (sb2.length() > 0) {
            sb2.append("|");
          }
          int corefClusterId = (gold)? m.goldCorefClusterID:m.corefClusterID;
          sb2.append("(").append(corefClusterId);
        }
        for(Mention m : mentionBeginEnd.get(i)){
          if (sb2.length() > 0) {
            sb2.append("|");
          }
          int corefClusterId = (gold)? m.goldCorefClusterID:m.corefClusterID;
          sb2.append("(").append(corefClusterId).append(")");
        }
        for(Mention m : mentionEndOnly.get(i)){
          if (sb2.length() > 0) {
            sb2.append("|");
          }
          int corefClusterId = (gold)? m.goldCorefClusterID:m.corefClusterID;
          sb2.append(corefClusterId).append(")");
        }
        if(sb2.length() == 0) sb2.append("-");

        //String[] columns = conllSentence.get(i);
        //for(int j = 0 ; j < columns.length-1 ; j++){
        //  String column = columns[j];
        //  sb.append(column).append("\t");
        //}
        sb.append(i+1).append("\t").append(sentence.get(i).get(TextAnnotation.class)).append("\t");
        sb.append(sb2).append("\n");
      }
      sb.append("\n");
    }

    sb.append("#end document").append("\n");
    //    sb.append("#end document ").append(docID).append("\n");

    writer.print(sb.toString());
    writer.flush();
  }

  /** Print raw document for analysis */
  private static void printRawDoc(Document document, boolean gold) throws FileNotFoundException {
    List<CoreMap> sentences = document.annotation.get(SentencesAnnotation.class);
    List<List<Mention>> allMentions;
    if(gold) allMentions = document.goldOrderedMentionsBySentence;
    else allMentions = document.predictedOrderedMentionsBySentence;
    //    String filename = document.annotation.get()

    StringBuilder doc = new StringBuilder();
    int previousOffset = 0;
    //Counter<Integer> mentionCount = new OpenAddressCounter<Integer>();
    for(List<Mention> l : allMentions) {
      for(Mention m : l){
        //mentionCount.incrementCount(m.goldCorefClusterID);
      }
    }

    for(int i = 0 ; i<sentences.size(); i++) {
      CoreMap sentence = sentences.get(i);
      List<Mention> mentions = allMentions.get(i);

      String[] tokens = sentence.get(TextAnnotation.class).split(" ");
      List<CoreLabel> t = sentence.get(TokensAnnotation.class);
      if(previousOffset+2 < t.get(0).get(CharacterOffsetBeginAnnotation.class)) {
        doc.append("\n");
      }
      previousOffset = t.get(t.size()-1).get(CharacterOffsetEndAnnotation.class);
      //Counter<Integer> startCounts = new OpenAddressCounter<Integer>();
      //Counter<Integer> endCounts = new OpenAddressCounter<Integer>();
      HashMap<Integer, Set<Mention>> endMentions = new HashMap<Integer, Set<Mention>>();
      for (Mention m : mentions) {
        //startCounts.incrementCount(m.startIndex);
        //endCounts.incrementCount(m.endIndex);
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
        /*for (int k = 0 ; k < startCounts.getCount(j) ; k++) {
          char lastChar = (doc.length() > 0)? doc.charAt(doc.length()-1):' ';
          if (lastChar != '[') doc.append(" ");
          doc.append("[");
        }*/
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
    logger.fine(document.annotation.get(DocIDAnnotation.class));
    if(gold) logger.fine("New DOC: (GOLD MENTIONS) ==================================================");
    else logger.fine("New DOC: (Predicted Mentions) ==================================================");
    logger.fine(doc.toString());
  }
  public static List<Pair<IntTuple, IntTuple>> getLinks(
      Map<Integer, CorefChain> result) {
    List<Pair<IntTuple, IntTuple>> links = new ArrayList<Pair<IntTuple, IntTuple>>();
    MentionComparator comparator = new MentionComparator();

    for(CorefChain c : result.values()) {
      List<CorefMention> s = c.getCorefMentions();
      for(CorefMention m1 : s){
        for(CorefMention m2 : s){
          if(comparator.compare(m1, m2)==1) links.add(new Pair<IntTuple, IntTuple>(m1.position, m2.position));
        }
      }
    }
    return links;
  }
}
