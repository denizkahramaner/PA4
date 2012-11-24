//
// StanfordCoreNLP -- a suite of NLP tools
// Copyright (c) 2009-2010 The Board of Trustees of
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

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;

import edu.stanford.nlp.jcoref.JointCorefClassifier;
import edu.stanford.nlp.jcoref.RuleBasedJointCorefSystem;
import edu.stanford.nlp.jcoref.dcoref.Dictionaries.Number;
import edu.stanford.nlp.jcoref.dcoref.Dictionaries.Person;
import edu.stanford.nlp.jcoref.dcoref.SieveCoreferenceSystem.Semantics;
import edu.stanford.nlp.jcoref.dcoref.sievepasses.JointArgumentMatch;
import edu.stanford.nlp.jcoref.docclustering.SimilarityVector;
import edu.stanford.nlp.ling.CoreAnnotations.CharacterOffsetBeginAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.CharacterOffsetEndAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.LemmaAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.NamedEntityTagAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.ParagraphAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.PartOfSpeechAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.SpeakerAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TextAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TokenBeginAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.UtteranceAnnotation;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.IndexedWord;
import edu.stanford.nlp.ling.RVFDatum;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.stats.ClassicCounter;
import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.trees.GrammaticalRelation;
import edu.stanford.nlp.trees.semgraph.SemanticGraph;
import edu.stanford.nlp.trees.semgraph.SemanticGraphCoreAnnotations.CollapsedDependenciesAnnotation;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.IntPair;
import edu.stanford.nlp.util.IntTuple;
import edu.stanford.nlp.util.Pair;

public class Document implements Serializable {

  private static final long serialVersionUID = -4139866807494603953L;

  public enum DocType { CONVERSATION, ARTICLE };

  /** The type of document: conversational or article */
  public DocType docType;

  /** Document annotation */
  public Annotation annotation;

  /** for conll shared task 2011  */
  public CoNLL2011DocumentReader.Document conllDoc;

  /** The list of gold mentions */
  public List<List<Mention>> goldOrderedMentionsBySentence;
  /** The list of predicted mentions */
  public List<List<Mention>> predictedOrderedMentionsBySentence;

  /** return the list of predicted mentions */
  public List<List<Mention>> getOrderedMentions() {
    return predictedOrderedMentionsBySentence;
  }

  /** Clusters for coreferent mentions */
  public Map<Integer, CorefCluster> corefClusters;

  /** Gold Clusters for coreferent mentions */
  public Map<Integer, CorefCluster> goldCorefClusters;

  /** All mentions in a document mentionID -> mention*/
  public Map<Integer, Mention> allPredictedMentions;
  public Map<Integer, Mention> allGoldMentions;

  /** Set of roles (in role apposition) in a document  */
  public Set<Mention> roleSet;

  /**
   * Position of each mention in the input matrix
   * Each mention occurrence with sentence # and position within sentence
   * (Nth mention, not Nth token)
   */
  public HashMap<Mention, IntTuple> positions;

  public final HashMap<IntPair, Mention> mentionheadPositions;

  /** List of gold links in a document by positions */
  private List<Pair<IntTuple,IntTuple>> goldLinks;

  /** UtteranceAnnotation -> String (speaker): mention ID or speaker string  */
  public Map<Integer, String> speakers;

  /** mention ID pair  */
  public Set<Pair<Integer, Integer>> speakerPairs;

  public int maxUtter;
  public int numParagraph;
  public int numSentences;

  /** Set of incompatible mention pairs */
  public Set<Pair<Integer, Integer>> incompatibles;

  /** 2nd order tf-idf sentence vectors */
  public List<SimilarityVector> sentVectors2ndOrder;

  public List<SimilarityVector> sentVectors1stOrder;

  /** cache for sentSimilarity */
  public Map<IntPair, Double> sent2ndOrderSimilarity;

  public Map<IntPair, Double> sent1stOrderSimilarity;

  /** cache for mention thesaurus similarity: mention ID pair is used */
  public Map<IntPair, Double> mentionSimilarity;

  /** cache for mention similarity (synonym/hypernym/sibling) in WN: mention ID pair is used */
  public Map<IntPair, Boolean> mentionSimilarInWN;

  /** cache for mention synonymy in WN: mention ID pair is used */
  public Set<IntPair> mentionSynonymInWN;

  /** for debug purpose */
  public Map<IntPair, String> errorLog;

  // # of arguments match counter between two clusters. (IntPair.first < IntPair.second)
  public Counter<IntPair> corefScore;

  public Document docForTracing = null;
  public List<IntPair> mergingList = new ArrayList<IntPair>();
  public Counter<Integer> calinskiScore = new ClassicCounter<Integer>();
  public HashMap<String, ClassicCounter<String>> verbMetaCentroid = new HashMap<String, ClassicCounter<String>>();
  public HashMap<String, ClassicCounter<String>> nominalMetaCentroid = new HashMap<String, ClassicCounter<String>>();

  public Counter<Integer> goldMentionCounterInCluster;

  public static class ValueComparator implements Comparator {
    Map base;
    public ValueComparator(Map base) {
      this.base = base;
    }
    public int compare(Object a, Object b) {
      if((Double)base.get(a) < (Double)base.get(b)) {
        return 1;
      } else if((Double)base.get(a) == (Double)base.get(b)) {
        return 0;
      } else {
        return -1;
      }
    }
  }
  
  /** @author Marta */
  public void setSingletons(){

    // Set token indices in doc
    EntityLifespan.setTokenIndices(this);

    for(List<Mention> sent : getOrderedMentions()){
      for(Mention men : sent){
        Pair<String, Integer> mention_key = new Pair<String, Integer>(conllDoc.documentID, men.headWord.get(TokenBeginAnnotation.class));
        if(SieveCoreferenceSystem.singletonProbs.containsKey(mention_key)
            && SieveCoreferenceSystem.singletonProbs.get(mention_key) < 0.2){
          men.isSingleton = true;
        }
      }
    }
  }

  public Document() {
    positions = new HashMap<Mention, IntTuple>();
    mentionheadPositions = new HashMap<IntPair, Mention>();
    roleSet = new HashSet<Mention>();
    corefClusters = new HashMap<Integer, CorefCluster>();
    goldCorefClusters = null;
    allPredictedMentions = new HashMap<Integer, Mention>();
    allGoldMentions = new HashMap<Integer, Mention>();
    speakers = new HashMap<Integer, String>();
    speakerPairs = new HashSet<Pair<Integer, Integer>>();
    incompatibles = new HashSet<Pair<Integer, Integer>>();
    errorLog = new HashMap<IntPair, String>();
    sentVectors2ndOrder = new ArrayList<SimilarityVector>();
    sentVectors1stOrder = new ArrayList<SimilarityVector>();
    sent2ndOrderSimilarity = new HashMap<IntPair, Double>();
    sent1stOrderSimilarity = new HashMap<IntPair, Double>();
    mentionSimilarity = new HashMap<IntPair, Double>();
    mentionSimilarInWN = new HashMap<IntPair, Boolean>();
    mentionSynonymInWN = new HashSet<IntPair>();
  }

  public Document(Annotation anno, List<List<Mention>> predictedMentions,
      List<List<Mention>> goldMentions, Dictionaries dict, Semantics semantics) {
    this();
    annotation = anno;
    numSentences = anno.get(SentencesAnnotation.class).size();
    predictedOrderedMentionsBySentence = predictedMentions;
    goldOrderedMentionsBySentence = goldMentions;
    if(goldMentions!=null) {
      findTwinMentions(Constants.STRICT_MENTION_BOUNDARY);
      // fill allGoldMentions
      for(List<Mention> l : goldOrderedMentionsBySentence) {
        for(Mention g : l) {
          allGoldMentions.put(g.mentionID, g);
        }
      }
    }
    // set original ID, initial coref clusters, paragraph annotation, mention positions
    initialize();
    processDiscourse(dict);
    printMentionDetection();
    for(CoreMap sent : anno.get(SentencesAnnotation.class)) {
      List<CoreLabel> sentence = sent.get(TokensAnnotation.class);
      sentVectors2ndOrder.add(SimilarityVector.get2ndOrderTfIdfSentenceVector(sentence, dict));
      sentVectors1stOrder.add(SimilarityVector.get1stOrderTfIdfSentenceVector(sentence, dict));
    }
    for(int i = 0 ; i < sentVectors2ndOrder.size(); i++) {
      for(int j=i+1 ; j < sentVectors2ndOrder.size(); j++) {
        IntPair pair = new IntPair(i,j);
        sent2ndOrderSimilarity.put(pair, SimilarityVector.getCosineSimilarity(sentVectors2ndOrder.get(i), sentVectors2ndOrder.get(j)));
        sent1stOrderSimilarity.put(pair, SimilarityVector.getCosineSimilarity(sentVectors1stOrder.get(i), sentVectors1stOrder.get(j)));
      }
    }
    for(Mention m1 : allPredictedMentions.values()) {
      for(Mention m2 : allPredictedMentions.values()) {
        if(m1.mentionID < m2.mentionID) {
          IntPair menPair = new IntPair(m1.mentionID, m2.mentionID);
          mentionSimilarity.put(menPair, SimilarityVector.getCosineSimilarity(m1.simVector, m2.simVector));
          mentionSimilarInWN.put(menPair, WordNet.similarInWN(m1, m2, semantics.wordnet, dict));
          if(WordNet.synonymInWN(m1, m2, semantics.wordnet, dict)) mentionSynonymInWN.add(menPair);
        }
      }
    }
  }
  /** Process discourse information */
  private void processDiscourse(Dictionaries dict) {
    docType = findDocType(dict);
    markQuotations(this.annotation.get(SentencesAnnotation.class), false);
    findSpeakers(dict);

    // find 'speaker mention' for each mention
    for(Mention m : allPredictedMentions.values()) {
      int utter = m.headWord.get(UtteranceAnnotation.class);
      int speakerMentionID;
      try{
        speakerMentionID = Integer.parseInt(m.headWord.get(SpeakerAnnotation.class));
        if(utter!=0) {
          speakerPairs.add(new Pair<Integer, Integer>(m.mentionID, speakerMentionID));
        }
      } catch (Exception e){
        // no mention found for the speaker
        // nothing to do
      }
      // set generic 'you' : e.g., you know in conversation
      if(docType!=DocType.ARTICLE && m.person==Person.YOU && m.endIndex < m.sentenceWords.size()-1
          && m.sentenceWords.get(m.endIndex).get(TextAnnotation.class).equalsIgnoreCase("know")) {
        m.generic = true;
      }
    }
  }
  /** Document initialize */
  private void initialize() {
    if(goldOrderedMentionsBySentence==null) assignOriginalID();
    setParagraphAnnotation();
    initializeCorefCluster();
  }

  /** initialize positions and corefClusters (put each mention in each CorefCluster) */
  private void initializeCorefCluster() {
    for(int i = 0; i < predictedOrderedMentionsBySentence.size(); i ++){
      for(int j = 0; j < predictedOrderedMentionsBySentence.get(i).size(); j ++){
        Mention m = predictedOrderedMentionsBySentence.get(i).get(j);
        allPredictedMentions.put(m.mentionID, m);

        IntTuple pos = new IntTuple(2);
        pos.set(0, i);
        pos.set(1, j);
        positions.put(m, pos);
        //        m.sentNum = i;  // now done in mention detection

        corefClusters.put(m.mentionID, new CorefCluster(m.mentionID, new HashSet<Mention>(Arrays.asList(m))));
        m.corefClusterID = m.mentionID;

        IntPair headPosition = new IntPair(i, m.headIndex);
        mentionheadPositions.put(headPosition, m);
      }
    }
  }

  /** Mark twin mentions in gold and predicted mentions */
  private void findTwinMentions(boolean strict){
    if(strict) findTwinMentionsStrict();
    else findTwinMentionsRelaxed();
  }

  /** Mark twin mentions: All mention boundaries should be matched */
  private void findTwinMentionsStrict(){
    for(int sentNum = 0; sentNum < goldOrderedMentionsBySentence.size(); sentNum++) {
      List<Mention> golds = goldOrderedMentionsBySentence.get(sentNum);
      List<Mention> predicts = predictedOrderedMentionsBySentence.get(sentNum);

      Map<IntPair, Mention> goldMentionPositions = new HashMap<IntPair, Mention>();
      for(Mention g : golds) {
        goldMentionPositions.put(new IntPair(g.startIndex, g.endIndex), g);
      }
      for(Mention p : predicts) {
        IntPair pos = new IntPair(p.startIndex, p.endIndex);
        if(goldMentionPositions.containsKey(pos)) {
          Mention g = goldMentionPositions.get(pos);
          if(p.additionalSpanStartIndex == g.additionalSpanStartIndex
              && p.additionalSpanEndIndex == g.additionalSpanEndIndex) {
            p.mentionID = g.mentionID;
            p.twinless = false;
            g.twinless = false;
            p.isEvent = g.isEvent;
            if(p.isEvent) p.findArguments();
          }
        }
      }
      // temp: for making easy to recognize twinless mention
      for(Mention p : predicts){
        if(p.twinless) p.mentionID += 10000;
      }
    }
  }

  /** Mark twin mentions: heads of the mentions are matched */
  private void findTwinMentionsRelaxed() {
    for(int sentNum = 0; sentNum < goldOrderedMentionsBySentence.size(); sentNum++) {
      List<Mention> golds = goldOrderedMentionsBySentence.get(sentNum);
      List<Mention> predicts = predictedOrderedMentionsBySentence.get(sentNum);

      Map<IntPair, Mention> goldMentionPositions = new HashMap<IntPair, Mention>();
      Map<Integer, LinkedList<Mention>> goldMentionHeadPositions = new HashMap<Integer, LinkedList<Mention>>();
      for(Mention g : golds) {
        goldMentionPositions.put(new IntPair(g.startIndex, g.endIndex), g);
        if(!goldMentionHeadPositions.containsKey(g.headIndex)) {
          goldMentionHeadPositions.put(g.headIndex, new LinkedList<Mention>());
        }
        goldMentionHeadPositions.get(g.headIndex).add(g);
      }

      List<Mention> remains = new ArrayList<Mention>();
      for(Mention p : predicts) {
        IntPair pos = new IntPair(p.startIndex, p.endIndex);
        if(goldMentionPositions.containsKey(pos)) {
          Mention g = goldMentionPositions.get(pos);
          if(p.additionalSpanStartIndex == g.additionalSpanStartIndex
              && p.additionalSpanEndIndex == g.additionalSpanEndIndex) {
            p.mentionID = g.mentionID;
            p.twinless = false;
            g.twinless = false;
            p.isEvent = g.isEvent;
            if(p.isEvent) p.findArguments();
            goldMentionHeadPositions.get(g.headIndex).remove(g);
            if(goldMentionHeadPositions.get(g.headIndex).size()==0) {
              goldMentionHeadPositions.remove(g.headIndex);
            }
          } else {
            remains.add(p);
          }
        }
        else remains.add(p);
      }
      for(Mention r : remains){
        if(goldMentionHeadPositions.containsKey(r.headIndex)) {
          Mention g = goldMentionHeadPositions.get(r.headIndex).poll();
          r.mentionID = g.mentionID;
          r.twinless = false;
          g.twinless = false;
          r.isEvent = g.isEvent;
          if(r.isEvent) r.findArguments();
          if(goldMentionHeadPositions.get(g.headIndex).size()==0) {
            goldMentionHeadPositions.remove(g.headIndex);
          }
        }
      }
    }
  }

  /** Set paragraph index */
  private void setParagraphAnnotation() {
    int paragraphIndex = 0;
    int previousOffset = -10;
    for(CoreMap sent : annotation.get(SentencesAnnotation.class)) {
      for(CoreLabel w : sent.get(TokensAnnotation.class)) {
        if(w.containsKey(CharacterOffsetBeginAnnotation.class)) {
          if(w.get(CharacterOffsetBeginAnnotation.class) > previousOffset+2) paragraphIndex++;
          w.set(ParagraphAnnotation.class, paragraphIndex);
          previousOffset = w.get(CharacterOffsetEndAnnotation.class);
        } else {
          w.set(ParagraphAnnotation.class, -1);
        }
      }
    }
    for(List<Mention> l : predictedOrderedMentionsBySentence) {
      for(Mention m : l){
        m.paragraph = m.headWord.get(ParagraphAnnotation.class);
      }
    }
    numParagraph = paragraphIndex;
  }

  /** Find document type: Conversation or article  */
  private DocType findDocType(Dictionaries dict) {
    boolean speakerChange = false;
    Set<Integer> discourseWithIorYou = new HashSet<Integer>();

    for(CoreMap sent : annotation.get(SentencesAnnotation.class)) {
      for(CoreLabel w : sent.get(TokensAnnotation.class)) {
        int utterIndex = w.get(UtteranceAnnotation.class);
        if(utterIndex!=0) speakerChange = true;
        if(speakerChange && utterIndex==0) return DocType.ARTICLE;
        if(dict.firstPersonPronouns.contains(w.get(TextAnnotation.class).toLowerCase())
            || dict.secondPersonPronouns.contains(w.get(TextAnnotation.class).toLowerCase())) {
          discourseWithIorYou.add(utterIndex);
        }
        if(maxUtter < utterIndex) maxUtter = utterIndex;
      }
    }
    if(!speakerChange) return DocType.ARTICLE;
    return DocType.CONVERSATION;  // in conversation, utter index keep increasing.
  }

  /** When there is no mentionID information (without gold annotation), assign mention IDs */
  protected void assignOriginalID(){
    List<List<Mention>> orderedMentionsBySentence = this.getOrderedMentions();
    boolean hasOriginalID = true;
    for(List<Mention> l : orderedMentionsBySentence){
      if (l.size()==0) continue;
      for(Mention m : l){
        if(m.mentionID == -1){
          hasOriginalID = false;
        }
      }
    }
    if(!hasOriginalID){
      int id = 0;
      for(List<Mention> l : orderedMentionsBySentence){
        for(Mention m : l){
          m.mentionID = id++;
        }
      }
    }
  }

  /** Extract gold coref cluster information */
  public void extractGoldCorefClusters(){
    goldCorefClusters = new HashMap<Integer, CorefCluster>();
    for (List<Mention> mentions : goldOrderedMentionsBySentence) {
      for (Mention m : mentions) {
        int id = m.goldCorefClusterID;
        if (id == -1) {
          throw new RuntimeException("No gold info");
        }
        CorefCluster c = goldCorefClusters.get(id);
        if (c == null) {
          goldCorefClusters.put(id, new CorefCluster());
        }
        c = goldCorefClusters.get(id);
        c.clusterID = id;
        c.corefMentions.add(m);
      }
    }
  }

  protected List<Pair<IntTuple, IntTuple>> getGoldLinks() {
    if(goldLinks==null) this.extractGoldLinks();
    return goldLinks;
  }

  /** Extract gold coref link information */
  protected void extractGoldLinks() {
    //    List<List<Mention>> orderedMentionsBySentence = this.getOrderedMentions();
    List<Pair<IntTuple, IntTuple>> links = new ArrayList<Pair<IntTuple,IntTuple>>();

    // position of each mention in the input matrix, by id
    HashMap<Integer, IntTuple> positions = new HashMap<Integer, IntTuple>();
    // positions of antecedents
    HashMap<Integer, List<IntTuple>> antecedents = new HashMap<Integer, List<IntTuple>>();
    for(int i = 0; i < goldOrderedMentionsBySentence.size(); i ++){
      for(int j = 0; j < goldOrderedMentionsBySentence.get(i).size(); j ++){
        Mention m = goldOrderedMentionsBySentence.get(i).get(j);
        int id = m.mentionID;
        IntTuple pos = new IntTuple(2);
        pos.set(0, i);
        pos.set(1, j);
        positions.put(id, pos);
        antecedents.put(id, new ArrayList<IntTuple>());
      }
    }

    for (List<Mention> mentions : goldOrderedMentionsBySentence) {
      for (Mention m : mentions) {
        int id = m.mentionID;
        IntTuple src = positions.get(id);

        assert (src != null);
        if (m.originalRef >= 0) {
          IntTuple dst = positions.get(m.originalRef);
          if (dst == null) {
            throw new RuntimeException("Cannot find gold mention with ID=" + m.originalRef);
          }

          // to deal with cataphoric annotation
          while (dst.get(0) > src.get(0) || (dst.get(0) == src.get(0) && dst.get(1) > src.get(1))) {
            Mention dstMention = goldOrderedMentionsBySentence.get(dst.get(0)).get(dst.get(1));
            m.originalRef = dstMention.originalRef;
            dstMention.originalRef = id;

            if (m.originalRef < 0) break;
            dst = positions.get(m.originalRef);
          }
          if (m.originalRef < 0) continue;

          // A B C: if A<-B, A<-C => make a link B<-C
          for (int k = dst.get(0); k <= src.get(0); k++) {
            for (int l = 0; l < goldOrderedMentionsBySentence.get(k).size(); l++) {
              if (k == dst.get(0) && l < dst.get(1)) continue;
              if (k == src.get(0) && l > src.get(1)) break;
              IntTuple missed = new IntTuple(2);
              missed.set(0, k);
              missed.set(1, l);
              if (links.contains(new Pair<IntTuple, IntTuple>(missed, dst))) {
                antecedents.get(id).add(missed);
                links.add(new Pair<IntTuple, IntTuple>(src, missed));
              }
            }
          }

          links.add(new Pair<IntTuple, IntTuple>(src, dst));

          assert (antecedents.get(id) != null);
          antecedents.get(id).add(dst);

          List<IntTuple> ants = antecedents.get(m.originalRef);
          assert (ants != null);
          for (IntTuple ant : ants) {
            antecedents.get(id).add(ant);
            links.add(new Pair<IntTuple, IntTuple>(src, ant));
          }
        }
      }
    }
    goldLinks = links;
  }

  /** set UtteranceAnnotation for quotations: default UtteranceAnnotation = 0 is given */
  private void markQuotations(List<CoreMap> results, boolean normalQuotationType) {
    boolean insideQuotation = false;
    for(CoreMap m : results) {
      for(CoreLabel l : m.get(TokensAnnotation.class)) {
        String w = l.get(TextAnnotation.class);

        boolean noSpeakerInfo = !l.containsKey(SpeakerAnnotation.class)
        || l.get(SpeakerAnnotation.class).equals("")
        || l.get(SpeakerAnnotation.class).startsWith("PER");

        if(w.equals("``")
            || (!insideQuotation && normalQuotationType && w.equals("\""))) {
          insideQuotation = true;
          maxUtter++;
          continue;
        } else if(w.equals("''")
            || (insideQuotation && normalQuotationType && w.equals("\""))) {
          insideQuotation = false;
        }
        if(insideQuotation) {
          l.set(UtteranceAnnotation.class, maxUtter);
        }
        if(noSpeakerInfo){
          l.set(SpeakerAnnotation.class, "PER"+l.get(UtteranceAnnotation.class));
        }
      }
    }
    if(maxUtter==0 && !normalQuotationType) markQuotations(results, true);
  }

  /** Speaker extraction */
  private void findSpeakers(Dictionaries dict) {
    if(Constants.USE_GOLD_SPEAKER_TAGS) {
      for(CoreMap sent : annotation.get(SentencesAnnotation.class)) {
        for(CoreLabel w : sent.get(TokensAnnotation.class)) {
          int utterIndex = w.get(UtteranceAnnotation.class);
          speakers.put(utterIndex, w.get(SpeakerAnnotation.class));
        }
      }
    } else {
      if(docType==DocType.CONVERSATION) findSpeakersInConversation(dict);
      else if (docType==DocType.ARTICLE) findSpeakersInArticle(dict);

      // set speaker info to annotation
      for(CoreMap sent : annotation.get(SentencesAnnotation.class)) {
        for(CoreLabel w : sent.get(TokensAnnotation.class)) {
          int utterIndex = w.get(UtteranceAnnotation.class);
          if(speakers.containsKey(utterIndex)) {
            w.set(SpeakerAnnotation.class, speakers.get(utterIndex));
          }
        }
      }
    }
  }
  private void findSpeakersInArticle(Dictionaries dict) {
    List<CoreMap> sentences = annotation.get(SentencesAnnotation.class);
    Pair<Integer, Integer> beginQuotation = new Pair<Integer, Integer>();
    Pair<Integer, Integer> endQuotation = new Pair<Integer, Integer>();
    boolean insideQuotation = false;
    int utterNum = -1;

    for (int i = 0 ; i < sentences.size(); i++) {
      List<CoreLabel> sent = sentences.get(i).get(TokensAnnotation.class);
      for(int j = 0 ; j < sent.size() ; j++) {
        int utterIndex = sent.get(j).get(UtteranceAnnotation.class);

        if(utterIndex != 0 && !insideQuotation) {
          utterNum = utterIndex;
          insideQuotation = true;
          beginQuotation.setFirst(i);
          beginQuotation.setSecond(j);
        } else if (utterIndex == 0 && insideQuotation) {
          insideQuotation = false;
          endQuotation.setFirst(i);
          endQuotation.setSecond(j);
          findQuotationSpeaker(utterNum, sentences, beginQuotation, endQuotation, dict);
        }
      }
    }
  }

  private void findQuotationSpeaker(int utterNum, List<CoreMap> sentences,
      Pair<Integer, Integer> beginQuotation, Pair<Integer, Integer> endQuotation, Dictionaries dict) {

    if(findSpeaker(utterNum, beginQuotation.first(), sentences, 0, beginQuotation.second(), dict))
      return ;

    if(findSpeaker(utterNum, endQuotation.first(), sentences, endQuotation.second(),
        sentences.get(endQuotation.first()).get(TokensAnnotation.class).size(), dict))
      return;

    if(beginQuotation.second() <= 1 && beginQuotation.first() > 0) {
      if(findSpeaker(utterNum, beginQuotation.first()-1, sentences, 0,
          sentences.get(beginQuotation.first()-1).get(TokensAnnotation.class).size(), dict))
        return ;
    }

    if(endQuotation.second() == sentences.get(endQuotation.first()).size()-1
        && sentences.size() > endQuotation.first()+1) {
      if(findSpeaker(utterNum, endQuotation.first()+1, sentences, 0,
          sentences.get(endQuotation.first()+1).get(TokensAnnotation.class).size(), dict))
        return ;
    }
  }

  private boolean findSpeaker(int utterNum, int sentNum, List<CoreMap> sentences,
      int startIndex, int endIndex, Dictionaries dict) {
    List<CoreLabel> sent = sentences.get(sentNum).get(TokensAnnotation.class);
    for(int i = startIndex ; i < endIndex ; i++) {
      if(sent.get(i).get(UtteranceAnnotation.class)!=0) continue;
      String lemma = sent.get(i).get(LemmaAnnotation.class);
      String word = sent.get(i).get(TextAnnotation.class);
      if(dict.reportVerb.contains(lemma)) {
        // find subject
        SemanticGraph dependency = sentences.get(sentNum).get(CollapsedDependenciesAnnotation.class);
        IndexedWord w = dependency.getNodeByWordPattern(word);

        if (w != null) {
          for(Pair<GrammaticalRelation,IndexedWord> child : dependency.childPairs(w)){
            if(child.first().getShortName().equals("nsubj")) {
              String subjectString = child.second().word();
              int subjectIndex = child.second().index();  // start from 1
              IntTuple headPosition = new IntTuple(2);
              headPosition.set(0, sentNum);
              headPosition.set(1, subjectIndex-1);
              String speaker;
              if(mentionheadPositions.containsKey(headPosition)) {
                speaker = Integer.toString(mentionheadPositions.get(headPosition).mentionID);
              } else {
                speaker = subjectString;
              }
              speakers.put(utterNum, speaker);
              return true;
            }
          }
        } else {
          SieveCoreferenceSystem.logger.warning("Cannot find node in dependency for word " + word);
        }
      }
    }
    return false;
  }

  private void findSpeakersInConversation(Dictionaries dict) {
    for(List<Mention> l : predictedOrderedMentionsBySentence) {
      for(Mention m : l){
        if(m.predicateNominatives == null) continue;
        for (Mention a : m.predicateNominatives){
          if(a.spanToString().toLowerCase().equals("i")) {
            speakers.put(m.headWord.get(UtteranceAnnotation.class), Integer.toString(m.mentionID));
          }
        }
      }
    }
    List<CoreMap> paragraph = new ArrayList<CoreMap>();
    int paragraphUtterIndex = 0;
    String nextParagraphSpeaker = "";
    int paragraphOffset = 0;
    for(CoreMap sent : annotation.get(SentencesAnnotation.class)) {
      int currentUtter = sent.get(TokensAnnotation.class).get(0).get(UtteranceAnnotation.class);
      if(paragraphUtterIndex!=currentUtter) {
        nextParagraphSpeaker = findParagraphSpeaker(paragraph, paragraphUtterIndex, nextParagraphSpeaker, paragraphOffset, dict);
        paragraphUtterIndex = currentUtter;
        paragraphOffset += paragraph.size();
        paragraph = new ArrayList<CoreMap>();
      }
      paragraph.add(sent);
    }
    findParagraphSpeaker(paragraph, paragraphUtterIndex, nextParagraphSpeaker, paragraphOffset, dict);
  }

  private String findParagraphSpeaker(List<CoreMap> paragraph,
      int paragraphUtterIndex, String nextParagraphSpeaker, int paragraphOffset, Dictionaries dict) {
    if(!speakers.containsKey(paragraphUtterIndex)) {
      if(!nextParagraphSpeaker.equals("")) {
        speakers.put(paragraphUtterIndex, nextParagraphSpeaker);
      } else {  // find the speaker of this paragraph (John, nbc news)
        CoreMap lastSent = paragraph.get(paragraph.size()-1);
        String speaker = "";
        boolean hasVerb = false;
        for(int i = 0 ; i < lastSent.get(TokensAnnotation.class).size() ; i++){
          CoreLabel w = lastSent.get(TokensAnnotation.class).get(i);
          String pos = w.get(PartOfSpeechAnnotation.class);
          String ner = w.get(NamedEntityTagAnnotation.class);
          if(pos.startsWith("V")) {
            hasVerb = true;
            break;
          }
          if(ner.startsWith("PER")) {
            IntTuple headPosition = new IntTuple(2);
            headPosition.set(0, paragraph.size()-1 + paragraphOffset);
            headPosition.set(1, i);
            if(mentionheadPositions.containsKey(headPosition)) {
              speaker = Integer.toString(mentionheadPositions.get(headPosition).mentionID);
            }
          }
        }
        if(!hasVerb && !speaker.equals("")) {
          speakers.put(paragraphUtterIndex, speaker);
        }
      }
    }
    return findNextParagraphSpeaker(paragraph, paragraphOffset, dict);
  }

  private String findNextParagraphSpeaker(List<CoreMap> paragraph, int paragraphOffset, Dictionaries dict) {
    CoreMap lastSent = paragraph.get(paragraph.size()-1);
    String speaker = "";
    for(CoreLabel w : lastSent.get(TokensAnnotation.class)) {
      if(w.get(LemmaAnnotation.class).equals("report") || w.get(LemmaAnnotation.class).equals("say")) {
        String word = w.get(TextAnnotation.class);
        SemanticGraph dependency = lastSent.get(CollapsedDependenciesAnnotation.class);
        IndexedWord t = dependency.getNodeByWordPattern(word);

        for(Pair<GrammaticalRelation,IndexedWord> child : dependency.childPairs(t)){
          if(child.first().getShortName().equals("nsubj")) {
            int subjectIndex = child.second().index();  // start from 1
            IntTuple headPosition = new IntTuple(2);
            headPosition.set(0, paragraph.size()-1 + paragraphOffset);
            headPosition.set(1, subjectIndex-1);
            if(mentionheadPositions.containsKey(headPosition)
                && mentionheadPositions.get(headPosition).nerString.startsWith("PER")) {
              speaker = Integer.toString(mentionheadPositions.get(headPosition).mentionID);
            }
          }
        }
      }
    }
    return speaker;
  }

  /** Check one mention is the speaker of the other mention */
  public static boolean isSpeaker(Mention m, Mention ant, Dictionaries dict) {

    if(!dict.firstPersonPronouns.contains(ant.spanToString().toLowerCase())
        || ant.number==Number.PLURAL || ant.sentNum!=m.sentNum) return false;

    int countQuotationMark = 0;
    for(int i = Math.min(m.headIndex, ant.headIndex)+1 ; i < Math.max(m.headIndex, ant.headIndex) ; i++) {
      String word = m.sentenceWords.get(i).get(TextAnnotation.class);
      if(word.equals("``") || word.equals("''")) countQuotationMark++;
    }
    if(countQuotationMark!=1) return false;

    IndexedWord w = m.dependency.getNodeByWordPattern(m.sentenceWords.get(m.headIndex).get(TextAnnotation.class));
    if(w== null) return false;

    for(Pair<GrammaticalRelation,IndexedWord> parent : m.dependency.parentPairs(w)){
      if(parent.first().getShortName().equals("nsubj")
          && dict.reportVerb.contains(parent.second().get(LemmaAnnotation.class))) {
        return true;
      }
    }
    return false;
  }

  private void printMentionDetection() {
    int foundGoldCount = 0;
    for(Mention g : allGoldMentions.values()) {
      if(!g.twinless) foundGoldCount++;
    }
    RuleBasedJointCorefSystem.logger.fine("# of found gold mentions: "+foundGoldCount + " / # of gold mentions: "+allGoldMentions.size());
    RuleBasedJointCorefSystem.logger.fine("gold mentions == ");
  }

  //  public void initializeArgMatchCounter() {
  //    argMatchCounterNominalCluster = new ClassicCounter<IntPair>();
  //    argMatchCounterVerbCluster = new ClassicCounter<IntPair>();
  //
  //    for(CorefCluster c1 : this.corefClusters.values()) {
  //      for(CorefCluster c2 : this.corefClusters.values()) {
  //        if(c1.clusterID >= c2.clusterID) continue;
  //        IntPair idPair = new IntPair(c1.clusterID, c2.clusterID);
  //        if(!c1.representative.isVerb && !c2.representative.isVerb) {
  //          this.argMatchCounterNominalCluster.incrementCount(idPair, Rules.entityArgMatchCount(c1, c2, this));
  //        } else {
  //          this.argMatchCounterVerbCluster.incrementCount(idPair, Rules.eventArgMatchCount(c1, c2, this));
  //        }
  //      }
  //    }
  //  }
  public void initializeArgMatchCounter(JointCorefClassifier jcc, Dictionaries dict, boolean first) {
    corefScore = new ClassicCounter<IntPair>();
    for(CorefCluster c : this.corefClusters.values()) {
      JointArgumentMatch.calculateCentroid(c, this, false);
      if(SieveCoreferenceSystem.trainJointCorefClassifier) {
        JointArgumentMatch.calculateCentroid(c, this, true);
      }
    }
    if(jcc.regressor==null) {
      for(CorefCluster c1 : this.corefClusters.values()) {
        for(CorefCluster c2 : this.corefClusters.values()) {
          if(c1.clusterID >= c2.clusterID || (!JointArgumentMatch.DOPRONOUN && (c1.representative.isPronominal() || c2.representative.isPronominal()))) continue;
//              ) continue;
          IntPair idPair = new IntPair(c1.clusterID, c2.clusterID);
          double sim = JointArgumentMatch.calculateSimilarity(this, c1, c2);
          if(sim > JointArgumentMatch.CUTOFF_THRESHOLD) {
            this.corefScore.setCount(idPair, sim);
          }
          if(SieveCoreferenceSystem.trainJointCorefClassifier) {
            double s = JointArgumentMatch.getMergeScore(this, c1, c2);
            if(Math.abs(s-0.5) > 0.0001){
              jcc.addData(new RVFDatum<Double, String>(JointArgumentMatch.getFeatures(this, c1, c2, true, dict), s));
            }
          }
        }
      }
    } else {
      for(CorefCluster c1 : this.corefClusters.values()) {
        for(CorefCluster c2 : this.corefClusters.values()) {
          if(c1.clusterID >= c2.clusterID || (!JointArgumentMatch.DOPRONOUN && (c1.representative.isPronominal() || c2.representative.isPronominal()))) continue;
//              ) continue;
          IntPair idPair = new IntPair(c1.clusterID, c2.clusterID);
          RVFDatum<Double, String> datum = new RVFDatum<Double, String>(JointArgumentMatch.getFeatures(this, c1, c2, false, dict));
          JointArgumentMatch.rawFeatures.put(idPair, datum);
          double sim = jcc.valueOf(datum);
          // TODO : temp
          if(!first) {
            if(datum.asFeaturesCounter().getCount("SRLAGREECOUNT") > 2) {
              sim = 1;
            }
          }
          if(sim > JointArgumentMatch.CUTOFF_THRESHOLD) {
            if((!c1.representative.isPronominal() && !c2.representative.isPronominal()) || datum.asFeaturesCounter().getCount("SRLAGREECOUNT")!=0) {
              this.corefScore.setCount(idPair, sim);
            }
          }
          if(SieveCoreferenceSystem.trainJointCorefClassifier) {
            double s = JointArgumentMatch.getMergeScore(this, c1, c2);
            if(Math.abs(s-0.5) > 0.0001){
              jcc.addData(new RVFDatum<Double, String>(JointArgumentMatch.getFeatures(this, c1, c2, true, dict), s));
            }
          }
        }
      }
    }
  }

  public static void copyClusterInfo(Document source, Document target) {
    for(Mention m1 : target.allPredictedMentions.values()) {
      for(Mention m2 : target.allPredictedMentions.values()) {
        if(m1.corefClusterID==m2.corefClusterID) continue;
        int sourceCluster1ID = source.allPredictedMentions.get(m1.mentionID).corefClusterID;
        int sourceCluster2ID = source.allPredictedMentions.get(m2.mentionID).corefClusterID;
        if(sourceCluster1ID == sourceCluster2ID) {
          if(sourceCluster1ID == m1.corefClusterID) {
            int removeID = m2.corefClusterID;
            CorefCluster.mergeClusters(target.corefClusters.get(m1.corefClusterID), target.corefClusters.get(m2.corefClusterID));
            target.corefClusters.remove(removeID);
          } else {
            int removeID = m1.corefClusterID;
            CorefCluster.mergeClusters(target.corefClusters.get(m2.corefClusterID), target.corefClusters.get(m1.corefClusterID));
            target.corefClusters.remove(removeID);
          }
        }
      }
    }
  }
}
