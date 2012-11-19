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

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.Set;

import edu.stanford.nlp.jcoref.JointCorefDocument;
import edu.stanford.nlp.jcoref.SwirlHelper;
import edu.stanford.nlp.jcoref.dcoref.Dictionaries.Animacy;
import edu.stanford.nlp.jcoref.dcoref.Mention.Argument;
import edu.stanford.nlp.jcoref.dcoref.SieveCoreferenceSystem.Semantics;
import edu.stanford.nlp.ling.CoreAnnotations.BeginIndexAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.IndexAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.LemmaAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.PartOfSpeechAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.UtteranceAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.ValueAnnotation;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.IndexedWord;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.trees.GrammaticalRelation;
import edu.stanford.nlp.trees.HeadFinder;
import edu.stanford.nlp.trees.SemanticHeadFinder;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.trees.TreeCoreAnnotations.TreeAnnotation;
import edu.stanford.nlp.trees.semgraph.SemanticGraph;
import edu.stanford.nlp.trees.semgraph.SemanticGraphCoreAnnotations.CollapsedDependenciesAnnotation;
import edu.stanford.nlp.trees.tregex.TregexMatcher;
import edu.stanford.nlp.trees.tregex.TregexPattern;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.IntPair;
import edu.stanford.nlp.util.Pair;

/**
 * Generic mention extractor from a corpus.
 *
 * @author Jenny Finkel
 * @author Mihai Surdeanu
 * @author Karthik Raghunathan
 * @author Heeyoung Lee
 * @author Sudarshan Rangarajan
 */
public class MentionExtractor {

  protected HeadFinder headFinder;

  protected String currentDocumentID;

  protected Dictionaries dictionaries;
  protected Semantics semantics;

  public CorefMentionFinder mentionFinder;
  protected StanfordCoreNLP stanfordProcessor;

  /** The maximum mention ID: for preventing duplicated mention ID assignment */
  protected int maxID = -1;

  public static final boolean VERBOSE = false;

  public MentionExtractor(Dictionaries dict, Semantics semantics) {
    this.headFinder = new SemanticHeadFinder();
    this.dictionaries = dict;
    this.semantics = semantics;
    this.mentionFinder = new RuleBasedCorefMentionFinder();  // Default
  }

  public void setMentionFinder(CorefMentionFinder mentionFinder)
  {
    this.mentionFinder = mentionFinder;
  }

  /**
   * Extracts the info relevant for coref from the next document in the corpus
   * @return List of mentions found in each sentence ordered according to the tree traversal.
   * @throws ClassNotFoundException
   */
  public Document nextDoc() throws ClassNotFoundException { return null; }

  public Document arrange(
      Annotation anno,
      List<List<CoreLabel>> words,
      List<Tree> trees,
      List<List<Mention>> unorderedMentions) {
    return arrange(anno, words, trees, unorderedMentions, null, false);
  }

  protected int getHeadIndex(Tree t) {
    Tree ht = t.headTerminal(headFinder);
    if(ht==null) return -1;  // temporary: a key which is matched to nothing
    CoreLabel l = (CoreLabel) ht.label();
    return (int) l.get(IndexAnnotation.class);
  }
  private String treeToKey(Tree t){
    int idx = getHeadIndex(t);
    String key = Integer.toString(idx) + ":" + t.toString();
    return key;
  }

  public Document arrange(
      Annotation anno,
      List<List<CoreLabel>> words,
      List<Tree> trees,
      List<List<Mention>> unorderedMentions,
      List<List<Mention>> unorderedGoldMentions,
      boolean doMergeLabels) {
    List<List<Mention>> predictedOrderedMentionsBySentence = arrange(anno, words, trees, unorderedMentions, doMergeLabels);
    List<List<Mention>> goldOrderedMentionsBySentence = null;
    if(unorderedGoldMentions != null) {
      goldOrderedMentionsBySentence = arrange(anno, words, trees, unorderedGoldMentions, doMergeLabels);
    }
    return new Document(anno, predictedOrderedMentionsBySentence, goldOrderedMentionsBySentence, dictionaries, this.semantics);
  }

  /**
   * Post-processes the extracted mentions. Here we set the Mention fields required for coref and order mentions by tree-traversal order.
   * @param words List of words in each sentence, in textual order
   * @param trees List of trees, one per sentence
   * @param unorderedMentions List of unordered, unprocessed mentions
   *                 Each mention MUST have startIndex and endIndex set!
   *                 Optionally, if scoring is desired, mentions must have mentionID and originalRef set.
   *                 All the other Mention fields are set here.
   * @return List of mentions ordered according to the tree traversal
   */
  public List<List<Mention>> arrange(
      Annotation anno,
      List<List<CoreLabel>> words,
      List<Tree> trees,
      List<List<Mention>> unorderedMentions,
      boolean doMergeLabels) {

    List<List<Mention>> orderedMentionsBySentence = new ArrayList<List<Mention>>();

    //
    // traverse all sentences and process each individual one
    //
    for(int sent = 0; sent < words.size(); sent ++){
      List<CoreLabel> sentence = words.get(sent);
      Tree tree = trees.get(sent);
      List<Mention> mentions = unorderedMentions.get(sent);
      HashMap<String, List<Mention>> mentionsToTrees = new HashMap<String, List<Mention>>();

      // merge the parse tree of the entire sentence with the sentence words
      if(doMergeLabels) mergeLabels(tree, sentence);

      //
      // set the surface information and the syntactic info in each mention
      // startIndex and endIndex MUST be set before!
      //
      for(Mention mention: mentions){
        mention.contextParseTree = tree;
        mention.sentenceWords = sentence;
        mention.originalSpan = new ArrayList<CoreLabel>(mention.sentenceWords.subList(mention.startIndex, mention.endIndex));
        if(!((CoreLabel)tree.label()).has(BeginIndexAnnotation.class)) tree.indexSpans(0);
        if(mention.headWord==null) {
          Tree headTree = ((RuleBasedCorefMentionFinder) mentionFinder).findSyntacticHead(mention, tree, sentence);
          mention.headWord = (CoreLabel)headTree.label();
          mention.headIndex = mention.headWord.get(IndexAnnotation.class) - 1;
        }
        if(mention.mentionSubTree==null) {
          // mentionSubTree = highest NP that has the same head
          Tree headTree = tree.getLeaves().get(mention.headIndex);
          if (headTree == null) { throw new RuntimeException("Missing head tree for a mention!"); }
          Tree t = headTree;
          while ((t = t.parent(tree)) != null) {
            if (t.headTerminal(headFinder) == headTree && t.value().equals("NP")) {
              mention.mentionSubTree = t;
            } else if(mention.mentionSubTree != null){
              break;
            }
          }
          if (mention.mentionSubTree == null) {
            mention.mentionSubTree = headTree;
          }
        }

        List<Mention> mentionsForTree = mentionsToTrees.get(treeToKey(mention.mentionSubTree));
        if(mentionsForTree == null){
          mentionsForTree = new ArrayList<Mention>();
          mentionsToTrees.put(treeToKey(mention.mentionSubTree), mentionsForTree);
        }
        mentionsForTree.add(mention);

        // generates all fields required for coref, such as gender, number, etc.
        mention.process(dictionaries, semantics, this);
      }

      //
      // Order all mentions in tree-traversal order
      //
      List<Mention> orderedMentions = new ArrayList<Mention>();
      orderedMentionsBySentence.add(orderedMentions);

      // extract all mentions in tree traversal order (alternative: tree.postOrderNodeList())
      for (Tree t : tree.preOrderNodeList()) {
        List<Mention> lm = mentionsToTrees.get(treeToKey(t));
        if(lm != null){
          for(Mention m: lm){
            orderedMentions.add(m);
          }
        }
      }

      //
      // find appositions, predicate nominatives, relative pronouns in this sentence
      //
      findSyntacticRelations(tree, orderedMentions);

      // find subject, directObject, indirectObject, possessor
      findArguments(orderedMentions);
    }
    return orderedMentionsBySentence;
  }

  private static void findArguments(List<Mention> orderedMentions) {
    for(Mention m : orderedMentions) {
      for(Mention arg : orderedMentions) {
        if(m==arg || m.arguments==null) continue;

        Argument arg1 = m.arguments.get(arg.headWord.get(LemmaAnnotation.class));
        if(arg1==null || arg1.relation==null) continue;  // TODO

        if(arg1.relation.size()==1) {
          String rel = arg1.relation.get(0).getRelation().toString();
          if(rel.equals("nsubj") && m.isEvent) {
            m.subject = arg;
            arg.predicate = m;
          } else if(rel.equals("dobj") && m.isEvent) {
            m.directObject = arg;
            arg.predicate = m;
          } else if(rel.equals("iobj") && m.isEvent) {
            m.indirectObject = arg;
            arg.predicate = m;
          } else if(rel.equals("poss")) {
            arg.possessor = m;
            arg.srlArgs.put("A0", m);
          }
        }
      }
    }
  }

  /**
   * Sets the label of the leaf nodes to be the CoreLabels in the given sentence
   * The original value() of the Tree nodes is preserved
   */
  public static void mergeLabels(Tree tree, List<CoreLabel> sentence) {
    int idx = 0;
    for (Tree t : tree.getLeaves()) {
      CoreLabel cl = sentence.get(idx ++);
      String value = t.value();
      cl.set(ValueAnnotation.class, value);
      t.setLabel(cl);
    }
    tree.indexLeaves();
  }

  static boolean inside(int i, Mention m) {
    return (i >= m.startIndex && i < m.endIndex);
  }

  /** Find syntactic relations (e.g., appositives) in a sentence */
  private void findSyntacticRelations(Tree tree, List<Mention> orderedMentions) {
    Set<Pair<Integer, Integer>> appos = new HashSet<Pair<Integer, Integer>>();
    findAppositions(tree, appos);
    markMentionRelation(orderedMentions, appos, "APPOSITION");

    Set<Pair<Integer, Integer>> preNomi = new HashSet<Pair<Integer, Integer>>();
    findPredicateNominatives(tree, preNomi);
    markMentionRelation(orderedMentions, preNomi, "PREDICATE_NOMINATIVE");

    Set<Pair<Integer, Integer>> relativePronounPairs = new HashSet<Pair<Integer, Integer>>();
    findRelativePronouns(tree, relativePronounPairs);
    markMentionRelation(orderedMentions, relativePronounPairs, "RELATIVE_PRONOUN");
  }

  /** Find syntactic pattern in a sentence by tregex */
  private void findTreePattern(Tree tree, String pattern, Set<Pair<Integer, Integer>> foundPairs) {
    try {
      TregexPattern tgrepPattern = TregexPattern.compile(pattern);
      TregexMatcher m = tgrepPattern.matcher(tree);
      while (m.find()) {
        Tree t = m.getMatch();
        Tree np1 = m.getNode("m1");
        Tree np2 = m.getNode("m2");
        Tree np3 = null;
        if(pattern.contains("m3")) np3 = m.getNode("m3");
        addFoundPair(np1, np2, t, foundPairs);
        if(np3!=null) addFoundPair(np2, np3, t, foundPairs);
      }
    } catch (Exception e) {
      e.printStackTrace();
      System.exit(0);
    }
  }

  private void addFoundPair(Tree np1, Tree np2, Tree t,
      Set<Pair<Integer, Integer>> foundPairs) {
    Tree head1 = np1.headTerminal(headFinder);
    Tree head2 = np2.headTerminal(headFinder);
    int h1 = ((CoreMap) head1.label()).get(IndexAnnotation.class) - 1;
    int h2 = ((CoreMap) head2.label()).get(IndexAnnotation.class) - 1;
    Pair<Integer, Integer> p = new Pair<Integer, Integer>(h1, h2);
    foundPairs.add(p);
  }

  private void findAppositions(Tree tree, Set<Pair<Integer, Integer>> appos) {
    String appositionPattern = "NP=m1 < (NP=m2 $.. (/,/ $.. NP=m3))";
    String appositionPattern2 = "NP=m1 < (NP=m2 $.. (/,/ $.. (SBAR < (WHNP < WP|WDT=m3))))";
    String appositionPattern3 = "/^NP(?:-TMP|-ADV)?$/=m1 < (NP=m2 $- /^,$/ $-- NP=m3 !$ CC|CONJP)";
    String appositionPattern4 = "/^NP(?:-TMP|-ADV)?$/=m1 < (PRN=m2 < (NP < /^NNS?|CD$/ $-- /^-LRB-$/ $+ /^-RRB-$/))";
    findTreePattern(tree, appositionPattern, appos);
    findTreePattern(tree, appositionPattern2, appos);
    findTreePattern(tree, appositionPattern3, appos);
    findTreePattern(tree, appositionPattern4, appos);
  }
  private void findPredicateNominatives(Tree tree, Set<Pair<Integer, Integer>> preNomi) {
    String predicateNominativePattern = "S < (NP=m1 $.. (VP < ((/VB/ < /^(am|are|is|was|were|'m|'re|'s|be)$/) $.. NP=m2)))";
    String predicateNominativePattern2 = "S < (NP=m1 $.. (VP < (VP < ((/VB/ < /^(be|been|being)$/) $.. NP=m2))))";
    //    String predicateNominativePattern2 = "NP=m1 $.. (VP < ((/VB/ < /^(am|are|is|was|were|'m|'re|'s|be)$/) $.. NP=m2))";
    findTreePattern(tree, predicateNominativePattern, preNomi);
    findTreePattern(tree, predicateNominativePattern2, preNomi);
  }
  private void findRelativePronouns(Tree tree, Set<Pair<Integer, Integer>> relativePronounPairs) {
    String relativePronounPattern = "NP < (NP=m1 $.. (SBAR < (WHNP < WP|WDT=m2)))";
    findTreePattern(tree, relativePronounPattern, relativePronounPairs);
  }
  private static void markMentionRelation(List<Mention> orderedMentions, Set<Pair<Integer, Integer>> foundPairs, String flag) {
    for(Mention m1 : orderedMentions){
      for(Mention m2 : orderedMentions){
        for(Pair<Integer, Integer> foundPair: foundPairs){
          if((foundPair.first == m1.headIndex && foundPair.second == m2.headIndex)){
            if(flag.equals("APPOSITION")) m2.addApposition(m1);
            else if(flag.equals("PREDICATE_NOMINATIVE")) m2.addPredicateNominatives(m1);
            else if(flag.equals("RELATIVE_PRONOUN")) m2.addRelativePronoun(m1);
            else throw new RuntimeException("check flag in markMentionRelation (dcoref/MentionExtractor.java)");
          }
        }
      }
    }
  }
  /**
   * Finds the tree the matches this span exactly
   * @param tree Leaves must be indexed!
   * @param first First element in the span (first position has offset 1)
   * @param last Last element included in the span (first position has offset 1)
   */
  public static Tree findExactMatch(Tree tree, int first, int last) {
    List<Tree> leaves = tree.getLeaves();
    int thisFirst = ((CoreMap) leaves.get(0).label()).get(IndexAnnotation.class);
    int thisLast = ((CoreMap) leaves.get(leaves.size() - 1).label()).get(IndexAnnotation.class);
    if(thisFirst == first && thisLast == last) {
      return tree;
    } else {
      Tree [] kids = tree.children();
      for(Tree k: kids){
        Tree t = findExactMatch(k, first, last);
        if(t != null) return t;
      }
    }
    return null;
  }

  /** Load Stanford Processor: skip unnecessary annotator */
  protected StanfordCoreNLP loadStanfordProcessor(Properties props) {
    boolean replicateCoNLL = Boolean.parseBoolean(props.getProperty(Constants.REPLICATECONLL_PROP, "false"));

    Properties pipelineProps = new Properties(props);
    StringBuilder annoSb = new StringBuilder("");
    if (!Constants.USE_GOLD_POS && !replicateCoNLL)  {
      annoSb.append("pos, lemma");
    } else {
      annoSb.append("lemma");
    }
    if(Constants.USE_TRUECASE) {
      annoSb.append(", truecase");
    }
    if (!Constants.USE_GOLD_NE && !replicateCoNLL)  {
      annoSb.append(", ner");
    }
    if (!Constants.USE_GOLD_PARSES && !replicateCoNLL)  {
      annoSb.append(", parse");
    }
    String annoStr = annoSb.toString();
    SieveCoreferenceSystem.logger.info("Ignoring specified annotators, using annotators=" + annoStr);
    pipelineProps.put("annotators", annoStr);
    return new StanfordCoreNLP(pipelineProps, false);
  }

  public static void initializeUtterance(List<CoreLabel> tokens) {
    for(CoreLabel l : tokens){
      l.set(UtteranceAnnotation.class, 0);
    }
  }

  public JointCorefDocument extractJointCorefDocument(Document doc) {
    if(doc==null) return null;

    List<List<CoreLabel>> sentences = new ArrayList<List<CoreLabel>>();
    List<Tree> trees = new ArrayList<Tree>();

    for (CoreMap sentence: doc.annotation.get(SentencesAnnotation.class)) {
      sentences.add(sentence.get(TokensAnnotation.class));
      Tree tree = sentence.get(TreeAnnotation.class);
      // TODO: need deep copy?
      //      Tree treeCopy = tree.treeSkeletonCopy();
      trees.add(tree);
    }

    JointCorefDocument document = arrange(doc, sentences, trees, doc.predictedOrderedMentionsBySentence, true);
    List<Map<Integer, Map<IntPair, String>>> srlInfo = SwirlHelper.readSRLOutput(SwirlHelper.conllSrlPath+"swirlOutput-"+document.docID+".txt");
    addSRLInfo(document, srlInfo);
    //    addSuperSense(document);

    return document;
  }

  public JointCorefDocument arrange(Document doc,
      List<List<CoreLabel>> words,
      List<Tree> trees,
      List<List<Mention>> unorderedMentions,
      boolean doMergeLabels) {

    Annotation anno = doc.annotation;
    List<List<Mention>> unorderedGoldMentions = doc.goldOrderedMentionsBySentence;

    List<List<Mention>> predictedOrderedMentionsBySentence = arrange(anno, words, trees, unorderedMentions, doMergeLabels);
    List<List<Mention>> goldOrderedMentionsBySentence = null;
    if(unorderedGoldMentions != null) {
      goldOrderedMentionsBySentence = arrange(anno, words, trees, unorderedGoldMentions, doMergeLabels);
    }
    return new JointCorefDocument(doc, predictedOrderedMentionsBySentence, goldOrderedMentionsBySentence, dictionaries, this.semantics);
  }

  /** add pre-processed srl info (including time, location, left and right mentions) */
  protected void addSRLInfo(JointCorefDocument document, List<Map<Integer, Map<IntPair, String>>> srlInfo) {
    List<List<Mention>> orderedMentions = document.getOrderedMentions();
    Map<IntPair, Mention> mentionheadPositions = document.mentionheadPositions;

    for(int sentNum = 0 ; sentNum < orderedMentions.size() ; sentNum++) {
      List<Mention> sentMentions = orderedMentions.get(sentNum);
      Map<Integer, Map<IntPair, String>> sentSrlInfo = (srlInfo.size() > sentNum)? srlInfo.get(sentNum) : new HashMap<Integer, Map<IntPair, String>>();

      for(int verbIdx : sentSrlInfo.keySet()) {
        findPropbankArg(mentionheadPositions.get(new IntPair(sentNum, verbIdx)), sentMentions, sentSrlInfo.get(verbIdx), document);
      }

      // if SRL misses TIME or LOCATION, use NER. if it is in enumeration, assign predicate of the mention with parent's
      Mention time = null;
      Mention location = null;
      for(Mention m : sentMentions) {
        if(m.nerString.equals("DATE") || m.nerString.equals("TIME")) time = m;
        if(m.nerString.startsWith("LOC")) location = m;
        if(m.parentInEnumeration!=null && !m.parentInEnumeration.srlPredicates.isEmpty()) {
          m.srlPredicates.putAll(m.parentInEnumeration.srlPredicates);
        }
      }
      if(time!=null || location != null) {
        for(Mention m : sentMentions) {
          CorefCluster mCluster = document.corefClusters.get(m.corefClusterID);
          if(m.time==null) m.time = time;
          if(m.location==null) m.location = location;
          if(!m.srlArgs.containsKey("AM-TMP")) {
            m.srlArgs.put("AM-TMP", time);
            if(!mCluster.srlRoles.containsKey("AM-TMP")) mCluster.srlRoles.put("AM-TMP", new HashSet<Mention>());
            mCluster.srlRoles.get("AM-TMP").add(time);
          }
          if(!m.srlArgs.containsKey("AM-LOC")) {
            m.srlArgs.put("AM-LOC", location);
            if(!mCluster.srlRoles.containsKey("AM-LOC")) mCluster.srlRoles.put("AM-LOC", new HashSet<Mention>());
            mCluster.srlRoles.get("AM-LOC").add(location);
          }
        }
      }
      // add left, right mention info
      for(int i = 0 ; i < sentMentions.size() ; i++) {
        Mention m = sentMentions.get(i);
        CorefCluster mCluster = document.corefClusters.get(m.corefClusterID);
        if(i > 0) {
          m.srlArgs.put("LEFT-MENTION", sentMentions.get(i-1));
          if(!mCluster.srlRoles.containsKey("LEFT-MENTION")) mCluster.srlRoles.put("LEFT-MENTION", new HashSet<Mention>());
          mCluster.srlRoles.get("LEFT-MENTION").add(sentMentions.get(i-1));
        }
        if(i < sentMentions.size()-1) {
          m.srlArgs.put("RIGHT-MENTION", sentMentions.get(i+1));
          if(!mCluster.srlRoles.containsKey("RIGHT-MENTION")) mCluster.srlRoles.put("RIGHT-MENTION", new HashSet<Mention>());
          mCluster.srlRoles.get("RIGHT-MENTION").add(sentMentions.get(i+1));
        }
      }

      //      findModifierArgs(sentMentions, document, sentNum);
    }
  }

  private void findModifierArgs(List<Mention> sentMentions, Document doc, int sentNum) {
    for(Mention m : sentMentions) {
      if(m.isVerb) continue;
      CoreMap sent = doc.annotation.get(SentencesAnnotation.class).get(sentNum);

      SemanticGraph collapsedDependencies = sent.get(CollapsedDependenciesAnnotation.class);
      IndexedWord mHead = collapsedDependencies.getNodeByIndexSafe(m.headWord.get(IndexAnnotation.class));
      if(mHead==null) continue;   // TODO check why it happens

      for(Mention m2 : sentMentions) {
        if(m==m2) continue;
        IndexedWord m2Head = collapsedDependencies.getNodeByIndexSafe(m2.headWord.get(IndexAnnotation.class));
        if(m2Head==null) continue;
        Collection<IndexedWord> parents = collapsedDependencies.getParents(m2Head);
        if(parents.contains(mHead)) {
          GrammaticalRelation reln = collapsedDependencies.reln(mHead, m2Head);
          String relnSpecific = reln.getSpecific();
          if(m2.animacy==Animacy.ANIMATE || (relnSpecific!=null && relnSpecific.equals("by"))) {
            m.srlArgs.put("A0", m2);
            m.srlArgs.put("A1", m2);
            //System.err.println("---------------------------------");
            //System.err.println("SENTENCE: "+sent.get(TextAnnotation.class));
            //System.err.println(m2.spanToString()+ "] is A0 arguments of ["+ m.spanToString());
          } else {
            m.srlArgs.put("A1", m2);
            m.srlArgs.put("A0", m2);
            //System.err.println("---------------------------------");
            //System.err.println("SENTENCE: "+sent.get(TextAnnotation.class));
            //System.err.println(m2.spanToString()+ "] is A1 arguments of ["+ m.spanToString());
          }
        }
      }
    }

  }

  private void findPropbankArg(Mention mention, List<Mention> sentMentions, Map<IntPair, String> mentionArgs, Document document) {
    if(mention==null) return;

    for(IntPair argSpan : mentionArgs.keySet()) {
      findCorrespondingMention(argSpan, mentionArgs.get(argSpan), mention, sentMentions, document);
    }

    // debug
    //    System.err.println("------------------------");
    //    System.err.println(mention.spanToString());
    //    System.err.println(Mention.sentenceWordsToString(mention));
    //    System.err.println("mention arguments(matched): "+mention.srlArgs);
    //    System.err.print("mention argSpan(from Swirl): ");
    //    for(IntPair pair : mentionArgs.keySet()) {
    //      StringBuilder sb = new StringBuilder(mentionArgs.get(pair));
    //      sb.append("=");
    //      for(int i = pair.get(0) ; i < pair.get(1) ; i++) {
    //        sb.append(mention.sentenceWords.get(i).get(TextAnnotation.class)).append(" ");
    //      }
    //      System.err.print(sb.append(", ").toString());
    //    }
    //    System.err.println();
  }

  private void findCorrespondingMention(IntPair argSpan, String role, Mention mention, List<Mention> sentMentions, Document document) {
    int headIdx = findHeadIndexOfArg(argSpan, mention);

    for(Mention m : sentMentions) {
      if(m.headIndex==headIdx) {
        if(role.equals("AM-TMP")) {
          for(Mention men : sentMentions) {
            if(!men.srlArgs.containsKey("AM-TMP")) men.srlArgs.put("AM-TMP", m);
            men.time = m;
          }
        }
        if(role.equals("AM-LOC")) {
          for(Mention men : sentMentions) {
            if(!men.srlArgs.containsKey("AM-LOC")) men.srlArgs.put("AM-LOC", m);
            men.location = m;
          }
        }

        mention.srlArgs.put(role, m);
        m.srlPredicates.put(mention, role);

        CorefCluster mentionCluster = document.corefClusters.get(mention.corefClusterID);
        CorefCluster mCluster = document.corefClusters.get(m.corefClusterID);

        if(!mentionCluster.srlRoles.containsKey(role)) mentionCluster.srlRoles.put(role, new HashSet<Mention>());
        if(!mCluster.srlPredicates.containsKey(role)) mCluster.srlPredicates.put(role, new HashSet<Mention>());

        mentionCluster.srlRoles.get(role).add(m);
        mCluster.srlPredicates.get(role).add(mention);
      }
    }
  }

  private int findHeadIndexOfArg(IntPair argSpan, Mention mention) {
    Tree sentence = mention.contextParseTree;
    List<Tree> leaves = sentence.getLeaves();

    String pos = mention.sentenceWords.get(argSpan.get(0)).get(PartOfSpeechAnnotation.class);

    Tree startNode = ((pos.equals("IN") || pos.equals("TO") || pos.equals("RP")) && leaves.size() > argSpan.get(0)+1)? leaves.get(argSpan.get(0)+1) : leaves.get(argSpan.get(0));
    Tree endNode = leaves.get(argSpan.get(1)-1);
    Tree node = startNode;
    int headIdx = argSpan.get(1);   // for single token mention

    while(node!=null) {
      if(node.dominates(endNode)) break;
      else {
        Tree parent = node.parent(sentence);
        headIdx = getHeadIndex(parent);
        if(headIdx > argSpan.get(0) && headIdx <= argSpan.get(1)) {
          node = parent;
        } else {
          headIdx = getHeadIndex(node);
          break;
        }
      }
    }
    return headIdx - 1;
  }

}
