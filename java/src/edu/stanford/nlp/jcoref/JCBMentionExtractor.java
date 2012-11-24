package edu.stanford.nlp.jcoref;

import java.util.ArrayList;
import java.util.List;
import java.util.Properties;

import edu.stanford.nlp.jcoref.SuperSenseTaggerReader.SSTToken;
import edu.stanford.nlp.jcoref.dcoref.Dictionaries;
import edu.stanford.nlp.jcoref.dcoref.Mention;
import edu.stanford.nlp.jcoref.dcoref.MentionExtractor;
import edu.stanford.nlp.jcoref.dcoref.SieveCoreferenceSystem.Semantics;
import edu.stanford.nlp.jcoref.dcoref.Document;
import edu.stanford.nlp.jcoref.dcoref.Constants;
import edu.stanford.nlp.jcoref.dcoref.RuleBasedCorefMentionFinder;
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TextAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.WordSenseAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.DocIDAnnotation;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.trees.TreeCoreAnnotations.TreeAnnotation;
import edu.stanford.nlp.trees.semgraph.SemanticGraphCoreAnnotations.BasicDependenciesAnnotation;
import edu.stanford.nlp.util.CollectionValuedMap;
import edu.stanford.nlp.util.CoreMap;

public class JCBMentionExtractor extends MentionExtractor {

  public JCBReader jcbReader;
  private final String jcbPath = "/scr/heeyoung/corpus/coref/jcoref/jcb_v0.2/";
  private final String sstPath = "/scr/heeyoung/corpus/coref/jcoref/jcb_dev/jcb_dev_tagged/";
  protected final boolean detectMentionsInAnnotatedSentencesOnly;

  public JCBMentionExtractor(Dictionaries dict, Properties props, Semantics semantics) {
    super(dict, semantics);
    stanfordProcessor = loadStanfordProcessor(props);
    String corpusPath = props.getProperty("JCBPath", jcbPath);
    jcbReader = new JCBReader(corpusPath, ".jcb");
    detectMentionsInAnnotatedSentencesOnly = Boolean.parseBoolean(props.getProperty("jcoref.annotatedSentenceOnly", "false"));
  }

  /** extract mentions. similar to old method nextDoc() */
  public JointCorefDocument extractMentions(JCBDocument metaDoc) {
    if(metaDoc==null) return null;

    stanfordProcessor.annotate(metaDoc.annotation);

    List<List<Mention>> allPredictedMentions = mentionFinder.extractPredictedMentions(metaDoc.annotation, jcbReader.maxGoldMentionID, dictionaries, semantics, metaDoc.goldMentions, detectMentionsInAnnotatedSentencesOnly);
    List<List<CoreLabel>> sentences = new ArrayList<List<CoreLabel>>();
    List<Tree> trees = new ArrayList<Tree>();

    for (CoreMap sentence: metaDoc.annotation.get(SentencesAnnotation.class)) {
      sentences.add(sentence.get(TokensAnnotation.class));
      Tree tree = sentence.get(TreeAnnotation.class);
      // TODO: need deep copy?
      //      Tree treeCopy = tree.treeSkeletonCopy();
      trees.add(tree);
    }

    JointCorefDocument document = arrange(metaDoc, sentences, trees, allPredictedMentions, true);
    addSRLInfo(document, metaDoc.srlInfo);
    //    addSuperSense(document);

    return document;
  }

/*
  public JointCorefDocument nextDoc() throws ClassNotFoundException {
    JCBDocument jcbDoc = jcbReader.nextDoc();
    return extractMentions(jcbDoc);
  }
*/

  public Document nextDoc() throws ClassNotFoundException {
    List<List<CoreLabel>> allWords = new ArrayList<List<CoreLabel>>();
    List<Tree> allTrees = new ArrayList<Tree>();

    JCBDocument jcbDoc = jcbReader.nextDoc();
    if (jcbDoc == null) {
      return null;
    }

    Annotation anno = jcbDoc.annotation;

    // Run pipeline
    stanfordProcessor.annotate(anno);

    // Add document id annotation
    anno.set(DocIDAnnotation.class, jcbDoc.docID);

    for (CoreMap sentence:anno.get(CoreAnnotations.SentencesAnnotation.class)) {
      allWords.add(sentence.get(CoreAnnotations.TokensAnnotation.class));
      allTrees.add(sentence.get(TreeAnnotation.class));
    }

    // Initialize gold mentions
    List<List<Mention>> allGoldMentions = jcbDoc.goldMentions;
//    List<List<Mention>> allGoldMentions = extractGoldMentions(jcbDoc);

    List<List<Mention>> allPredictedMentions;
    if (Constants.USE_GOLD_MENTIONS) {
      allPredictedMentions = allGoldMentions;
    } else if (Constants.USE_GOLD_MENTION_BOUNDARIES) {
      allPredictedMentions = ((RuleBasedCorefMentionFinder) mentionFinder).filterPredictedMentions(allGoldMentions, anno, dictionaries);
    } else {
      allPredictedMentions = mentionFinder.extractPredictedMentions(anno, jcbReader.maxGoldMentionID, dictionaries, semantics);
    }

    Document doc = arrange(anno, allWords, allTrees, allPredictedMentions, allGoldMentions, true);
    
    // Set dependencies
    for(List<Mention> l : allPredictedMentions){  
      for(Mention men : l){
        men.dependency = doc.annotation.get(SentencesAnnotation.class).get(men.sentNum).get(BasicDependenciesAnnotation.class);
      }      
    }
    return doc;
  }


  /**
   * add pre-processed supersense tag
   * TODO: the code is messy due to tokenization issue
   * */
  private void addSuperSense(JointCorefDocument document) {
    List<List<SSTToken>> sstInfo = SuperSenseTaggerReader.readSSTOutput(sstPath+"topic"+document.docID.split("-")[0]+".tags");
    List<CoreMap> sentences = document.annotation.get(SentencesAnnotation.class);

    for(int sentNum = 0; sentNum < sstInfo.size(); sentNum++) {
      List<SSTToken> sentSST = sstInfo.get(sentNum);
      List<CoreLabel> sentence = sentences.get(sentNum).get(TokensAnnotation.class);

      for(int sstIdx = 0, sentIdx = 0 ; sstIdx < sentSST.size() && sentIdx < sentence.size() ; sstIdx++, sentIdx++) {
        String sstText = sentSST.get(sstIdx).text;
        String sentText = sentence.get(sentIdx).get(TextAnnotation.class);
        if(sstText.equals("(") || sstText.equals("[")) sstText = "-LRB-";
        else if(sstText.equals(")") || sstText.equals("]")) sstText = "-RRB-";
        else if(sstText.equals("{")) sstText = "-LCB-";
        else if(sstText.equals("}")) sstText = "-RCB-";
        if(sstText.contains("/")) sstText = sstText.replace("/", "\\/");
        if(sstText.startsWith(sentText) || sentText.startsWith(sstText)) sentence.get(sentIdx).set(WordSenseAnnotation.class, sentSST.get(sstIdx).supersense);
        else {
          if(sstText.startsWith("'") && sentText.equals("`")){
            sentText = "'";
          }
          System.err.println("tokenization mismatch: "+sstText+", "+sentText);
        }

        // fix tokenization mismatch
        if(!sstText.equals(sentText)) {
          if(sstText.contains(sentText)) {
            while(!sstText.equals(sentText)) {
              sentIdx++;
              if(sentIdx >= sentence.size()) {
                System.err.println("tokenization mismatch: "+sstText+", "+sentText);
                break;
              }
              sentText += sentence.get(sentIdx).get(TextAnnotation.class);
            }
          } else if(sentText.contains(sstText)) {
            while(!sentText.equals(sstText)) {
              sstIdx++;
              if(sstIdx >= sentSST.size()) {
                System.err.println("tokenization mismatch: "+sstText+", "+sentText);
                break;
              }
              sstText += sentSST.get(sstIdx).text;
            }
          }

        }
      }
    }
  }

  public JointCorefDocument arrange(JCBDocument metaDoc,
      List<List<CoreLabel>> words,
      List<Tree> trees,
      List<List<Mention>> unorderedMentions,
      boolean doMergeLabels) {

    Annotation anno = metaDoc.annotation;
    List<List<Mention>> unorderedGoldMentions = metaDoc.goldMentions;

    List<List<Mention>> predictedOrderedMentionsBySentence = arrange(anno, words, trees, unorderedMentions, doMergeLabels);
    List<List<Mention>> goldOrderedMentionsBySentence = null;
    if(unorderedGoldMentions != null) {
      goldOrderedMentionsBySentence = arrange(anno, words, trees, unorderedGoldMentions, doMergeLabels);
    }
    return new JointCorefDocument(metaDoc, predictedOrderedMentionsBySentence, goldOrderedMentionsBySentence, dictionaries, this.semantics);
  }

  //  public JointCorefDocument arrange(String docID, Annotation annotation, List<List<CoreLabel>> sentences,
  //      List<Tree> trees, List<List<Mention>> allPredictedMentions) {
  //    return arrange(docID, annotation, sentences, trees, allPredictedMentions, null, false);
  //  }
  
  /*public List<List<Mention>> extractGoldMentions(JCBDocument jcbDoc) {
    List<CoreMap> sentences = jcbDoc.annotation.get(CoreAnnotations.SentencesAnnotation.class);
    List<List<Mention>> allGoldMentions = new ArrayList<List<Mention>>();
    CollectionValuedMap<String,CoreMap> corefChainMap = jcbDoc.getCorefChainMap();
    for (int i = 0; i < sentences.size(); i++) {
      allGoldMentions.add(new ArrayList<Mention>());
    }
    int maxCorefClusterId = -1;
    for (String corefIdStr:corefChainMap.keySet()) {
      int id = Integer.parseInt(corefIdStr);
      if (id > maxCorefClusterId) {
        maxCorefClusterId = id;
      }
    }
    int newMentionID = maxCorefClusterId + 1;
    for (String corefIdStr:corefChainMap.keySet()) {
      int id = Integer.parseInt(corefIdStr);
      int clusterMentionCnt = 0;
      for (CoreMap m:corefChainMap.get(corefIdStr)) {
        clusterMentionCnt++;
        Mention mention = new Mention();

        mention.goldCorefClusterID = id;
        if (clusterMentionCnt == 1) {
          // First mention in cluster
          mention.mentionID = id;
          mention.originalRef = -1;
        } else {
          mention.mentionID = newMentionID;
          mention.originalRef = id;
          newMentionID++;
        }
        if(maxID < mention.mentionID) maxID = mention.mentionID;
        int sentIndex = m.get(CoreAnnotations.SentenceIndexAnnotation.class);
        CoreMap sent = sentences.get(sentIndex);
        mention.startIndex = m.get(CoreAnnotations.TokenBeginAnnotation.class) - sent.get(CoreAnnotations.TokenBeginAnnotation.class);
        mention.endIndex = m.get(CoreAnnotations.TokenEndAnnotation.class) - sent.get(CoreAnnotations.TokenBeginAnnotation.class);

        // will be set by arrange
        mention.originalSpan = m.get(CoreAnnotations.TokensAnnotation.class);

        // Mention dependency is collapsed dependency for sentence
        mention.dependency = sentences.get(sentIndex).get(CollapsedDependenciesAnnotation.class);

        allGoldMentions.get(sentIndex).add(mention);
      }
    }
    return allGoldMentions;
  }*/
  
}
