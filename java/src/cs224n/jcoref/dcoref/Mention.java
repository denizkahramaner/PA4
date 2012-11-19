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

import java.io.IOException;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import net.didion.jwnl.data.POS;
import net.didion.jwnl.data.Synset;
import edu.stanford.nlp.jcoref.dcoref.Dictionaries.Animacy;
import edu.stanford.nlp.jcoref.dcoref.Dictionaries.Gender;
import edu.stanford.nlp.jcoref.dcoref.Dictionaries.MentionType;
import edu.stanford.nlp.jcoref.dcoref.Dictionaries.Number;
import edu.stanford.nlp.jcoref.dcoref.Dictionaries.Person;
import edu.stanford.nlp.jcoref.dcoref.SieveCoreferenceSystem.Semantics;
import edu.stanford.nlp.jcoref.dcoref.WordNet.WNsynset;
import edu.stanford.nlp.jcoref.docclustering.SimilarityVector;
import edu.stanford.nlp.ling.CoreAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.EntityTypeAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.IndexAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.LemmaAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.NamedEntityTagAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.PartOfSpeechAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.SpeakerAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TextAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.UtteranceAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.ValueAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.WordSenseAnnotation;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.IndexedWord;
import edu.stanford.nlp.trees.GrammaticalRelation;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.trees.semgraph.SemanticGraph;
import edu.stanford.nlp.trees.semgraph.SemanticGraphEdge;
import edu.stanford.nlp.trees.tregex.TregexMatcher;
import edu.stanford.nlp.trees.tregex.TregexPattern;
import edu.stanford.nlp.util.IntPair;
import edu.stanford.nlp.util.Pair;
import edu.stanford.nlp.util.StringUtils;

/**
 * One mention for the SieveCoreferenceSystem
 * @author Jenny Finkel, Karthik Raghunathan, Heeyoung Lee
 */
public class Mention implements CoreAnnotation<Mention>, Serializable {

  private static final long serialVersionUID = -7524485803945717057L;


  public MentionType mentionType;
  public Number number;
  public edu.stanford.nlp.jcoref.dcoref.Dictionaries.Gender gender;
  public Animacy animacy;
  public Person person;
  public String headString;
  public String nerString;
  private String spanToString;
  public String lowercaseSpan;
  public String acronym;
  public int speakerID = -1;

  public int startIndex;
  public int endIndex;
  public int headIndex;
  public int mentionID = -1;
  public int originalRef = -1;
  public String originalDocID = "";

  // for representing mentions with discontinuous spans. e.g., check A into
  // originalSpan should have the headword
  public int additionalSpanStartIndex = -1;
  public int additionalSpanEndIndex = -1;
  public List<CoreLabel> additionalSpan = null;

  public int goldCorefClusterID = -1;
  public int corefClusterID = -1;
  public int sentNum = -1;
  public int utter = -1;
  public int paragraph = -1;
  public boolean isSubject;
  public boolean isDirectObject;
  public boolean isIndirectObject;
  public boolean isPrepositionObject;
  public IndexedWord dependingVerb;
  public boolean twinless = true;
  public boolean generic = false;   // generic pronoun or generic noun (bare plurals)
  public boolean isEvent = false;
  public boolean isVerb = false;
  public boolean isReport = false;
  public boolean isPassive = false;
  public boolean isEnumeration = false;
  public Mention parentInEnumeration = null;
  public List<Mention> mentionsInEnumeration = new ArrayList<Mention>();
  public Mention time = null;
  public Mention location = null;
  public boolean isSingleton = false;  // For the entity lifespan project (Marta, Marie)

  public List<CoreLabel> sentenceWords;
  public List<CoreLabel> originalSpan;

  public Tree mentionSubTree;
  public Tree contextParseTree;
  public CoreLabel headWord;
  public SemanticGraph dependency;
  public Set<String> dependents = new HashSet<String>();
  public List<String> preprocessedTerms;
  public WNsynset synsets = null;
  public SimilarityVector simVector = null;

  /** Set of other mentions in the same sentence that are syntactic appositions to this */
  public Set<Mention> appositions = null;
  public Set<Mention> predicateNominatives = null;
  public Set<Mention> relativePronouns = null;

  public Map<String, Argument> arguments;

  /** argument of a verb mention */
  public Mention subject = null;
  public Mention directObject = null;
  public Mention indirectObject = null;

  /**
   * propbank style arguments
   * e.g., The earthquake attacked the island. -> (attacked).srlArgs has (A0, The earthquake) and (A1, the island).
   * */
  public Map<String, Mention> srlArgs = new HashMap<String, Mention>();

  /**
   * propbank style predicates of this mention and its role
   * e.g., The earthquake attacked the island. -> (The earthquake).srlPredicates has (attacked, A0).
   *   */
  public Map<Mention, String> srlPredicates = new HashMap<Mention, String>();

  public Mention predicate = null;

  /** argument of a mention: A's B */
  public Mention possessor = null;

  /** Argument of Event Mention */
  public static class Argument implements Serializable {
    private static final long serialVersionUID = 4337409712595535359L;
    CoreLabel word;
    IntPair wordPosition;
    //    Mention mention;
    List<SemanticGraphEdge> relation;

    public Argument (CoreLabel w, IntPair pos, List<SemanticGraphEdge> rel) {
      word = w;
      wordPosition = pos;
      relation = rel;
    }
    @Override
    public String toString() {
      return word.get(TextAnnotation.class)+"->"+relation;
    }
  }

  public Mention() {
  }
  public Mention(int mentionID, int startIndex, int endIndex, SemanticGraph dependency){
    this.mentionID = mentionID;
    this.startIndex = startIndex;
    this.endIndex = endIndex;
    this.dependency = dependency;
    setSpanToString();
    findArguments();
  }
  public Mention(int mentionID, int startIndex, int endIndex, int sentNum, SemanticGraph dependency, List<CoreLabel> mentionSpan){
    this.mentionID = mentionID;
    this.startIndex = startIndex;
    this.endIndex = endIndex;
    this.sentNum = sentNum;
    this.dependency = dependency;
    this.originalSpan = mentionSpan;
    setSpanToString();
    findArguments();
  }
  public Mention(int mentionID, int startIndex, int endIndex, int sentNum, SemanticGraph dependency, List<CoreLabel> mentionSpan, Tree mentionTree){
    this.mentionID = mentionID;
    this.startIndex = startIndex;
    this.endIndex = endIndex;
    this.sentNum = sentNum;
    this.dependency = (dependency==null)? new SemanticGraph() : dependency;
    this.originalSpan = mentionSpan;
    this.mentionSubTree = mentionTree;
    setSpanToString();
    findArguments();
  }

  public Mention(int mentionID, int beginIdx, int endIdx, int sentNum,
      SemanticGraph dependency, ArrayList<CoreLabel> mentionSpan, Tree t, boolean isEvent) {
    this(mentionID, beginIdx, endIdx, sentNum, dependency, mentionSpan, t);
    this.isEvent = isEvent;
  }

  protected void findArguments() {
    arguments = new HashMap<String, Argument>();
    IndexedWord thisNode = dependency.getNodeByIndexSafe(startIndex+1);
    if(thisNode != null) {
      for(int i = 1 ; i <= dependency.size() ; i++) {
        IndexedWord w = dependency.getNodeByIndexSafe(i);
        if(thisNode==w || w==null) continue;    // TODO
        List<SemanticGraphEdge> path = dependency.getShortestUndirectedPathEdges(thisNode, w);
        arguments.put(w.get(LemmaAnnotation.class), new Argument(w, new IntPair(sentNum, w.get(IndexAnnotation.class)), path));
      }
    }
  }

  public Class<Mention> getType() {  return Mention.class; }

  public boolean isPronominal() {
    return mentionType == MentionType.PRONOMINAL;
  }

  @Override
  public String toString() {
    //    return headWord.toString();
    if(spanToString==null) {
      StringBuilder sb = new StringBuilder();
      sb.append("mention ").append(this.mentionID).append(" in cluster ").append(this.goldCorefClusterID);
      return sb.toString();
    }
    else return spanToString;
  }

  public String spanToString() {
    return spanToString;
  }

  private void setSpanToString() {
    StringBuilder os = new StringBuilder();
    for(int i = 0; i < originalSpan.size(); i ++){
      if(i > 0) os.append(" ");
      os.append(originalSpan.get(i).get(TextAnnotation.class));
    }
    this.spanToString = os.toString();
    this.lowercaseSpan = this.spanToString.toLowerCase();
  }

  /** Set attributes of a mention:
   * head string, mention type, NER label, Number, Gender, Animacy
   * @throws IOException */
  public void process(Dictionaries dict, Semantics semantics, MentionExtractor mentionExtractor) {
    setSpanToString();
    setHeadString();
    setType(dict);
    setNERString();
    List<String> mStr = getMentionString();
    setNumber(dict, getNumberCount(dict, mStr));
    setGender(dict, getGenderCount(dict, mStr));
    setAnimacy(dict);
    setPerson(dict);
    setDiscourse();
    setAcronym();

    if(semantics!=null) setSemantics(dict, semantics, mentionExtractor);
    setEventInfo(dict);
    simVector = SimilarityVector.get2ndOrderTfIdfSentenceVector(this.originalSpan, dict);
  }

  private void setAcronym() {
    char[] span = this.spanToString.toCharArray();
    StringBuilder sb = new StringBuilder();
    for(char c : span){
      if(Character.isUpperCase(c)){
        sb.append(c);
      }
    }
    this.acronym = sb.toString();
    if(this.spanToString.contains(this.acronym) && !this.headString.equalsIgnoreCase(this.acronym)
        && !this.headString.equals("'") && !this.headString.equals("'s")) this.acronym = "";
  }
  private void setEventInfo(Dictionaries dict) {
    if(dict.reportVerb.contains(this.headWord.get(LemmaAnnotation.class))
        || dict.reportNoun.contains(this.headWord.get(LemmaAnnotation.class))) {
      this.isReport = true;
    }
    if(headWord.get(PartOfSpeechAnnotation.class).equals("VBN")
        && headIndex > 0 && sentenceWords.get(headIndex-1).get(LemmaAnnotation.class).equals("be")) {
      this.isPassive = true;
    }
  }
  private List<String> getMentionString() {
    List<String> mStr = new ArrayList<String>();
    for(CoreLabel l : this.originalSpan) {
      mStr.add(l.get(TextAnnotation.class).toLowerCase());
      if(l==this.headWord) break;   // remove words after headword
    }
    return mStr;
  }
  private int[] getNumberCount(Dictionaries dict, List<String> mStr) {
    int len = mStr.size();
    if(len > 1) {
      for(int i = 0 ; i < len-1 ; i++) {
        if(dict.genderNumber.containsKey(mStr.subList(i, len))) return dict.genderNumber.get(mStr.subList(i, len));
      }

      // find converted string with ! (e.g., "dr. martin luther king jr. boulevard" -> "! boulevard")
      List<String> convertedStr = new ArrayList<String>();
      convertedStr.add("!");
      convertedStr.add(mStr.get(len-1));
      if(dict.genderNumber.containsKey(convertedStr)) return dict.genderNumber.get(convertedStr);
    }
    if(dict.genderNumber.containsKey(mStr.subList(len-1, len))) return dict.genderNumber.get(mStr.subList(len-1, len));

    return null;
  }
  private int[] getGenderCount(Dictionaries dict, List<String> mStr) {
    int len = mStr.size();
    char firstLetter = headWord.get(TextAnnotation.class).charAt(0);
    if(len > 1 && Character.isUpperCase(firstLetter) && nerString.startsWith("PER")) {
      int firstNameIdx = len-2;
      String secondToLast = mStr.get(firstNameIdx);
      if(firstNameIdx > 1 && (secondToLast.length()==1 || (secondToLast.length()==2 && secondToLast.endsWith(".")))) {
        firstNameIdx--;
      }

      for(int i = 0 ; i <= firstNameIdx ; i++){
        if(dict.genderNumber.containsKey(mStr.subList(i, len))) return dict.genderNumber.get(mStr.subList(i, len));
      }

      // find converted string with ! (e.g., "dr. martin luther king jr. boulevard" -> "dr. !")
      List<String> convertedStr = new ArrayList<String>();
      convertedStr.add(mStr.get(firstNameIdx));
      convertedStr.add("!");
      if(dict.genderNumber.containsKey(convertedStr)) return dict.genderNumber.get(convertedStr);

      if(dict.genderNumber.containsKey(mStr.subList(firstNameIdx, firstNameIdx+1))) return dict.genderNumber.get(mStr.subList(firstNameIdx, firstNameIdx+1));
    }

    if(dict.genderNumber.containsKey(mStr.subList(len-1, len))) return dict.genderNumber.get(mStr.subList(len-1, len));
    return null;
  }
  private void setDiscourse() {

    try {
      if(headWord.containsKey(SpeakerAnnotation.class)) {
        speakerID = Integer.parseInt(headWord.get(SpeakerAnnotation.class));
      }
    } catch (Exception e) {
      speakerID = -1;   // default value
    }

    utter = headWord.get(UtteranceAnnotation.class);

    Pair<IndexedWord, String> verbDependency = findDependentVerb(this);
    String dep = verbDependency.second();
    dependingVerb = verbDependency.first();

    isSubject = false;
    isDirectObject = false;
    isIndirectObject = false;
    isPrepositionObject = false;

    if(dep==null) {
      return ;
    } else if(dep.equals("nsubj") || dep.equals("csubj")) {
      isSubject = true;
    } else if(dep.equals("dobj")){
      isDirectObject = true;
    } else if(dep.equals("iobj")){
      isIndirectObject = true;
    } else if(dep.equals("pobj")){
      isPrepositionObject = true;
    }
  }

  private void setPerson(Dictionaries dict) {
    // only do for pronoun
    if(!this.isPronominal()) person = Person.UNKNOWN;

    if(dict.firstPersonPronouns.contains(lowercaseSpan)) {
      if(number==Number.SINGULAR) person = Person.I;
      else if(number==Number.PLURAL) person = Person.WE;
      else person = Person.UNKNOWN;
    } else if(dict.secondPersonPronouns.contains(lowercaseSpan)) {
      person = Person.YOU;
    } else if(dict.thirdPersonPronouns.contains(lowercaseSpan)) {
      if(gender==Gender.MALE && number==Number.SINGULAR) person = Person.HE;
      else if(gender==Gender.FEMALE && number==Number.SINGULAR) person = Person.SHE;
      else if((gender==Gender.NEUTRAL || animacy==Animacy.INANIMATE) && number==Number.SINGULAR) person = Person.IT;
      else if(number==Number.PLURAL) person = Person.THEY;
      else person = Person.UNKNOWN;
    } else {
      person = Person.UNKNOWN;
    }
  }

  private void setSemantics(Dictionaries dict, Semantics semantics, MentionExtractor mentionExtractor) {
    if(this.synsets!=null) return;
    if(this.headWord.get(PartOfSpeechAnnotation.class).startsWith("V")) {
      String head = this.headWord.get(LemmaAnnotation.class);
      if(this.sentenceWords.size() > this.headIndex+1) {
        CoreLabel nextToken = this.sentenceWords.get(this.headIndex+1);
        String nextWord = nextToken.get(LemmaAnnotation.class);
        String query = head+"_"+nextWord;   // query for phrase verb
        Synset[] syns = semantics.wordnet.wn.synsetsOf(query, POS.VERB);
        if(syns!=null) synsets = new WNsynset(syns, semantics.wordnet.wn.getNominalizationOfVerb(query), query);
      }
      if(synsets == null) {
        Synset[] syns = semantics.wordnet.wn.synsetsOf(head, POS.VERB);
        if(syns!=null) synsets = new WNsynset(syns, semantics.wordnet.wn.getNominalizationOfVerb(head), head);
      }
      else if(synsets.synsets.length == 0) {
        throw new RuntimeException("Mention.java: synsets length is 0");
      }
    } else {
      preprocessedTerms = this.preprocessSearchTerm();

      if(dict.statesAbbreviation.containsKey(this.spanToString())) {  // states abbreviations
        preprocessedTerms = new ArrayList<String>();
        preprocessedTerms.add(dict.statesAbbreviation.get(this.spanToString()));
      }
      synsets = semantics.wordnet.findSynset(preprocessedTerms);

      if(this.isPronominal()) return;
    }
    if(synsets!=null && synsets.synsets==null) {
      throw new RuntimeException("Mention.java: synsets.synsets == null");
    }
  }
  private void setType(Dictionaries dict) {
    if (headWord.has(EntityTypeAnnotation.class)){    // ACE gold mention type
      if (headWord.get(EntityTypeAnnotation.class).equals("PRO")) {
        mentionType = MentionType.PRONOMINAL;
      } else if (headWord.get(EntityTypeAnnotation.class).equals("NAM")) {
        mentionType = MentionType.PROPER;
      } else {
        mentionType = MentionType.NOMINAL;
      }
    } else {    // MUC
      if(!headWord.has(NamedEntityTagAnnotation.class)) {   // temporary fix
        mentionType = MentionType.NOMINAL;
        SieveCoreferenceSystem.logger.finest("no NamedEntityTagAnnotation: "+headWord);
      } else if (headWord.get(PartOfSpeechAnnotation.class).startsWith("PRP")
          || (originalSpan.size() == 1 && headWord.get(NamedEntityTagAnnotation.class).equals("O")
              && (dict.allPronouns.contains(headString) || dict.relativePronouns.contains(headString) ))) {
        mentionType = MentionType.PRONOMINAL;
      } else if (!headWord.get(NamedEntityTagAnnotation.class).equals("O") || headWord.get(PartOfSpeechAnnotation.class).startsWith("NNP")) {
        mentionType = MentionType.PROPER;
      } else {
        mentionType = MentionType.NOMINAL;
      }
    }
  }

  private void setGender(Dictionaries dict, int[] genderNumberCount) {
    gender = Gender.UNKNOWN;
    if (mentionType == MentionType.PRONOMINAL) {
      if (dict.malePronouns.contains(headString)) {
        gender = Gender.MALE;
      } else if (dict.femalePronouns.contains(headString)) {
        gender = Gender.FEMALE;
      }
    } else {
      if(Constants.USE_GENDER_LIST){
        // Bergsma list
        if(gender == Gender.UNKNOWN)  {
          if(dict.maleWords.contains(headString)) {
            gender = Gender.MALE;
            SieveCoreferenceSystem.logger.finest("[Bergsma List] New gender assigned:\tMale:\t" +  headString);
          }
          else if(dict.femaleWords.contains(headString))  {
            gender = Gender.FEMALE;
            SieveCoreferenceSystem.logger.finest("[Bergsma List] New gender assigned:\tFemale:\t" +  headString);
          }
          else if(dict.neutralWords.contains(headString))   {
            gender = Gender.NEUTRAL;
            SieveCoreferenceSystem.logger.finest("[Bergsma List] New gender assigned:\tNeutral:\t" +  headString);
          }
        }
      }
      if(genderNumberCount!=null && this.number!=Number.PLURAL){
        double male = genderNumberCount[0];
        double female = genderNumberCount[1];
        double neutral = genderNumberCount[2];

        if(male*0.5 > female+neutral && male > 2) this.gender = Gender.MALE;
        else if (female*0.5 > male+neutral && female > 2) this.gender = Gender.FEMALE;
        else if (neutral*0.5 > male+female && neutral > 2) this.gender = Gender.NEUTRAL;
      }
    }
  }

  private void setNumber(Dictionaries dict, int[] genderNumberCount) {
    if(this.isEnumeration) {
      number = Number.PLURAL;
      return;
    }
    if (mentionType == MentionType.PRONOMINAL) {
      if (dict.pluralPronouns.contains(headString)) {
        number = Number.PLURAL;
      } else if (dict.singularPronouns.contains(headString)) {
        number = Number.SINGULAR;
      } else {
        number = Number.UNKNOWN;
      }
    } else if(! nerString.equals("O") && mentionType!=MentionType.NOMINAL){
      if(! (nerString.equals("ORGANIZATION") || nerString.startsWith("ORG"))){
        number = Number.SINGULAR;
      } else {
        // ORGs can be both plural and singular
        number = Number.UNKNOWN;
      }
    } else {
      String tag = headWord.get(PartOfSpeechAnnotation.class);
      if (tag.startsWith("N") && tag.endsWith("S")) {
        number = Number.PLURAL;
      } else if (tag.startsWith("N")) {
        number = Number.SINGULAR;
      } else {
        number = Number.UNKNOWN;
      }
    }

    if(mentionType != MentionType.PRONOMINAL) {
      if(Constants.USE_NUMBER_LIST){
        if(number == Number.UNKNOWN){
          if(dict.singularWords.contains(headString)) {
            number = Number.SINGULAR;
            SieveCoreferenceSystem.logger.finest("[Bergsma] Number set to:\tSINGULAR:\t" + headString);
          }
          else if(dict.pluralWords.contains(headString))  {
            number = Number.PLURAL;
            SieveCoreferenceSystem.logger.finest("[Bergsma] Number set to:\tPLURAL:\t" + headString);
          }
        }
      }

      String enumerationPattern = "NP < (NP=tmp $.. (/,|CC/ $.. NP))";
      TregexPattern tgrepPattern = TregexPattern.compile(enumerationPattern);
      TregexMatcher m = tgrepPattern.matcher(this.mentionSubTree);
      while (m.find()) {
        //        Tree t = m.getMatch();
        if(this.mentionSubTree==m.getNode("tmp")
            && lowercaseSpan.contains(" and ")) {
          number = Number.PLURAL;
        }
      }
    }
  }

  private void setAnimacy(Dictionaries dict) {
    if (mentionType == MentionType.PRONOMINAL) {
      if (dict.animatePronouns.contains(headString)) {
        animacy = Animacy.ANIMATE;
      } else if (dict.inanimatePronouns.contains(headString)) {
        animacy = Animacy.INANIMATE;
      } else {
        animacy = Animacy.UNKNOWN;
      }
    } else if (nerString.equals("PERSON") || nerString.startsWith("PER")) {
      animacy = Animacy.ANIMATE;
    } else if (nerString.equals("LOCATION")|| nerString.startsWith("LOC")) {
      animacy = Animacy.INANIMATE;
    } else if (nerString.equals("MONEY")) {
      animacy = Animacy.INANIMATE;
    } else if (nerString.equals("NUMBER")) {
      animacy = Animacy.INANIMATE;
    } else if (nerString.equals("PERCENT")) {
      animacy = Animacy.INANIMATE;
    } else if (nerString.equals("DATE")) {
      animacy = Animacy.INANIMATE;
    } else if (nerString.equals("TIME")) {
      animacy = Animacy.INANIMATE;
    } else if (nerString.equals("MISC")) {
      animacy = Animacy.UNKNOWN;
    } else if (nerString.startsWith("VEH")) {
      animacy = Animacy.UNKNOWN;
    } else if (nerString.startsWith("FAC")) {
      animacy = Animacy.INANIMATE;
    } else if (nerString.startsWith("GPE")) {
      animacy = Animacy.INANIMATE;
    } else if (nerString.startsWith("WEA")) {
      animacy = Animacy.INANIMATE;
    } else if (nerString.startsWith("ORG")) {
      animacy = Animacy.INANIMATE;
    } else {
      animacy = Animacy.UNKNOWN;
    }
    if(mentionType != MentionType.PRONOMINAL) {
      if(Constants.USE_ANIMACY_LIST){
        // Better heuristics using DekangLin:
        if(animacy == Animacy.UNKNOWN)  {
          if(dict.animateWords.contains(headString))  {
            animacy = Animacy.ANIMATE;
            SieveCoreferenceSystem.logger.finest("Assigned Dekang Lin animacy:\tANIMATE:\t" + headString);
          }
          else if(dict.inanimateWords.contains(headString)) {
            animacy = Animacy.INANIMATE;
            SieveCoreferenceSystem.logger.finest("Assigned Dekang Lin animacy:\tINANIMATE:\t" + headString);
          }
        }
      }
    }
  }

  private static final String [] commonNESuffixes = {
    "Corp", "Co", "Inc", "Ltd"
  };
  private static boolean knownSuffix(String s) {
    if(s.endsWith(".")) s = s.substring(0, s.length() - 1);
    for(String suff: commonNESuffixes){
      if(suff.equalsIgnoreCase(s)){
        return true;
      }
    }
    return false;
  }

  private void setHeadString() {
    this.headString = headWord.get(TextAnnotation.class).toLowerCase();
    if(headWord.has(NamedEntityTagAnnotation.class)) {
      // make sure that the head of a NE is not a known suffix, e.g., Corp.
      int start = headIndex - startIndex;
      while(start >= 0){
        String head = originalSpan.get(start).get(TextAnnotation.class).toLowerCase();
        if(knownSuffix(head) == false){
          this.headString = head;
          break;
        } else {
          start --;
        }
      }
    }
  }

  private void setNERString() {
    if(headWord.has(EntityTypeAnnotation.class)){ // ACE
      if(headWord.has(NamedEntityTagAnnotation.class) && headWord.get(EntityTypeAnnotation.class).equals("NAM")){
        this.nerString = headWord.get(NamedEntityTagAnnotation.class);
      } else {
        this.nerString = "O";
      }
    }
    else{ // MUC
      if (headWord.has(NamedEntityTagAnnotation.class)) {
        this.nerString = headWord.get(NamedEntityTagAnnotation.class);
      } else {
        this.nerString = "O";
      }
    }
  }

  public boolean sameSentence(Mention m) {
    return m.sentenceWords == sentenceWords;
  }

  private static boolean included(CoreLabel small, List<CoreLabel> big) {
    if(small.tag().equals("NNP")){
      for(CoreLabel w: big){
        if(small.word().equals(w.word()) ||
            small.word().length() > 2 && w.word().startsWith(small.word())){
          return true;
        }
      }
    }
    return false;
  }

  protected boolean headsAgree(Mention m) {
    // we allow same-type NEs to not match perfectly, but rather one could be included in the other, e.g., "George" -> "George Bush"
    if(! nerString.equals("O") && ! m.nerString.equals("O") && nerString.equals(m.nerString) &&
        (included(headWord, m.originalSpan) || included(m.headWord, originalSpan)))
      return true;
    return headString.equals(m.headString);
  }

  public boolean numbersAgree(Mention m){
    return numbersAgree(m, false);
  }
  private boolean numbersAgree(Mention m, boolean strict) {
    if(strict){
      return number == m.number;
    }
    else return number == Number.UNKNOWN ||
    m.number == Number.UNKNOWN ||
    number == m.number;
  }

  public boolean gendersAgree(Mention m){
    return gendersAgree(m, false);
  }
  public boolean gendersAgree(Mention m, boolean strict) {
    if(strict){
      return gender == m.gender;
    }

    else return gender == Gender.UNKNOWN ||
    m.gender == Gender.UNKNOWN ||
    gender == m.gender;
  }

  public boolean animaciesAgree(Mention m){
    return animaciesAgree(m, false);
  }
  public boolean animaciesAgree(Mention m, boolean strict) {
    if(strict){
      return animacy == m.animacy;
    }

    else return animacy == Animacy.UNKNOWN ||
    m.animacy == Animacy.UNKNOWN ||
    animacy == m.animacy;
  }

  public boolean entityTypesAgree(Mention m, Dictionaries dict){
    return entityTypesAgree(m, dict, false);
  }

  public boolean entityTypesAgree(Mention m, Dictionaries dict, boolean strict) {
    if(strict) return nerString.equals(m.nerString);
    else{
      if (isPronominal()) {
        if(nerString.contains("-") || m.nerString.contains("-")){ //for ACE with gold NE
          if (m.nerString.equals("O")) {
            return true;
          } else if (m.nerString.startsWith("ORG")) {
            return dict.organizationPronouns.contains(headString);
          } else if (m.nerString.startsWith("PER")) {
            return dict.personPronouns.contains(headString);
          } else if (m.nerString.startsWith("LOC")) {
            return dict.locationPronouns.contains(headString);
          } else if (m.nerString.startsWith("GPE")) {
            return dict.GPEPronouns.contains(headString);
          } else if (m.nerString.startsWith("VEH") || m.nerString.startsWith("FAC")|| m.nerString.startsWith("WEA")) {
            return dict.facilityVehicleWeaponPronouns.contains(headString);
          } else {
            return false;
          }
        }
        else{  // ACE w/o gold NE or MUC
          if (m.nerString.equals("O")) {
            return true;
          } else if (m.nerString.equals("MISC")) {
            return true;
          } else if (m.nerString.equals("ORGANIZATION")) {
            return dict.organizationPronouns.contains(headString);
          } else if (m.nerString.equals("PERSON")) {
            return dict.personPronouns.contains(headString);
          } else if (m.nerString.equals("LOCATION")) {
            return dict.locationPronouns.contains(headString);
          } else if (m.nerString.equals("DATE") || m.nerString.equals("TIME") ) {
            return dict.dateTimePronouns.contains(headString);
          } else if (m.nerString.equals("MONEY") || m.nerString.equals("PERCENT") || m.nerString.equals("NUMBER")) {
            return dict.moneyPercentNumberPronouns.contains(headString);
          } else {
            return false;
          }
        }
      }
      return nerString.equals("O") ||
      m.nerString.equals("O") ||
      nerString.equals(m.nerString);
    }
  }

  /**
   * Verifies if this mention's tree is dominated by the tree of the given mention
   */
  public boolean includedIn(Mention m) {
    if (!m.sameSentence(this)) {
      return false;
    }
    if(this.startIndex < m.startIndex || this.endIndex > m.endIndex) return false;
    for (Tree t : m.mentionSubTree.subTrees()) {
      if (t == mentionSubTree) {
        return true;
      }
    }
    return false;
  }

  /**
   * Detects if the mention and candidate antecedent agree on all attributes respectively.
   * @param potentialAntecedent
   * @return true if all attributes agree between both mention and candidate, else false.
   */
  public boolean attributesAgree(Mention potentialAntecedent, Dictionaries dict){
    return (this.animaciesAgree(potentialAntecedent) &&
        this.entityTypesAgree(potentialAntecedent, dict) &&
        this.gendersAgree(potentialAntecedent) &&
        this.numbersAgree(potentialAntecedent));
  }

  /** Find apposition */
  public void addApposition(Mention m) {
    if(appositions == null) appositions = new HashSet<Mention>();
    appositions.add(m);
  }

  /** Check apposition */
  public boolean isApposition(Mention m) {
    if(appositions != null && appositions.contains(m)) return true;
    return false;
  }
  /** Find predicate nominatives */
  public void addPredicateNominatives(Mention m) {
    if(predicateNominatives == null) predicateNominatives = new HashSet<Mention>();
    predicateNominatives.add(m);
  }

  /** Check predicate nominatives */
  public boolean isPredicateNominatives(Mention m) {
    if(predicateNominatives != null && predicateNominatives.contains(m)) return true;
    return false;
  }

  /** Find relative pronouns */
  public void addRelativePronoun(Mention m) {
    if(relativePronouns == null) relativePronouns = new HashSet<Mention>();
    relativePronouns.add(m);
  }

  /** Check relative pronouns */
  public boolean isRelativePronoun(Mention m) {
    if(relativePronouns != null && relativePronouns.contains(m)) return true;
    return false;
  }

  public boolean isAcronym(Mention m) {
    if(m.spanToString.equals(acronym)) return true;
    return false;
  }

  public boolean isRoleAppositive(Mention m, Dictionaries dict) {
    String thisString = this.spanToString();
    if(this.isPronominal() || dict.allPronouns.contains(lowercaseSpan)) return false;
    if(!m.nerString.startsWith("PER") && !m.nerString.equals("O")) return false;
    if(!this.nerString.startsWith("PER") && !this.nerString.equals("O")) return false;
    if(!sameSentence(m) || !m.spanToString().startsWith(thisString)) return false;
    if(m.spanToString().contains("'") || m.spanToString().contains(" and ")) return false;
    if(!animaciesAgree(m) || this.animacy == Animacy.INANIMATE
        || this.gender == Gender.NEUTRAL || m.gender == Gender.NEUTRAL
        || !this.numbersAgree(m) ) return false;
    if(dict.demonymSet.contains(lowercaseSpan)
        || dict.demonymSet.contains(m.lowercaseSpan)) return false;
    return true;
  }

  public boolean isDemonym(Mention m, Dictionaries dict){
    String thisString = this.lowercaseSpan;
    String antString = m.lowercaseSpan;
    if(thisString.startsWith("the ") || thisString.startsWith("The ")) {
      thisString = thisString.substring(4);
    }
    if(antString.startsWith("the ") || antString.startsWith("The ")) antString = antString.substring(4);

    if(dict.statesAbbreviation.containsKey(m.spanToString()) && dict.statesAbbreviation.get(m.spanToString()).equals(this.spanToString())
        || dict.statesAbbreviation.containsKey(this.spanToString()) && dict.statesAbbreviation.get(this.spanToString()).equals(m.spanToString())) return true;

    if(dict.demonyms.get(thisString)!=null){
      if(dict.demonyms.get(thisString).contains(antString)) return true;
    } else if(dict.demonyms.get(antString)!=null){
      if(dict.demonyms.get(antString).contains(thisString)) return true;
    }
    return false;
  }

  /** Check whether later mention has incompatible modifier */
  public boolean haveIncompatibleModifier(Mention ant) {
    if(!ant.headString.equalsIgnoreCase(this.headString)) return false;   // only apply to same head mentions
    boolean thisHasExtra = false;
    int lengthThis = this.originalSpan.size();
    int lengthM = ant.originalSpan.size();
    Set<String> thisWordSet = new HashSet<String>();
    Set<String> antWordSet = new HashSet<String>();
    Set<String> locationModifier = new HashSet<String>(Arrays.asList("east", "west", "north", "south",
        "eastern", "western", "northern", "southern", "upper", "lower"));

    for (int i=0; i< lengthThis ; i++){
      String w1 = this.originalSpan.get(i).get(TextAnnotation.class).toLowerCase();
      String pos1 = this.originalSpan.get(i).get(PartOfSpeechAnnotation.class);
      if(!(pos1.startsWith("N") || pos1.startsWith("JJ") || pos1.equals("CD")
          || pos1.startsWith("V")) || w1.equalsIgnoreCase(this.headString)) continue;
      thisWordSet.add(w1);
    }
    for (int j=0 ; j < lengthM ; j++){
      String w2 = ant.originalSpan.get(j).get(TextAnnotation.class).toLowerCase();
      antWordSet.add(w2);
    }
    for (String w : thisWordSet){
      if(!antWordSet.contains(w)) thisHasExtra = true;
    }
    boolean hasLocationModifier = false;
    for(String l : locationModifier){
      if(antWordSet.contains(l) && !thisWordSet.contains(l)) {
        hasLocationModifier = true;
      }
    }
    return (thisHasExtra || hasLocationModifier);
  }

  /** Find which mention appears first in a document */
  public boolean appearEarlierThan(Mention m){
    if(this.sentNum < m.sentNum) return true;
    else if(this.sentNum > m.sentNum) return false;
    else {
      if(this.startIndex < m.startIndex) return true;
      else if(this.startIndex > m.startIndex) return false;
      else{
        if(this.endIndex > m.endIndex) return true;
        else return false;
      }
    }
  }

  /** Remove any clause after headword except enumeration */
  public String removePhraseAfterHead(){
    if(this.isEnumeration) return this.spanToString();
    String removed ="";
    int posComma = -1;
    int posWH = -1;
    for(int i = 0 ; i < this.originalSpan.size() ; i++){
      String pos = this.originalSpan.get(i).get(PartOfSpeechAnnotation.class);
      if(posComma == -1 && pos.equals(",")) posComma = this.startIndex + i;
      if(posWH == -1 && pos.startsWith("W")) posWH = this.startIndex + i;
    }
    if(posComma!=-1 && this.headIndex < posComma){
      StringBuilder os = new StringBuilder();
      for(int i = 0; i < posComma-this.startIndex; i++){
        if(i > 0) os.append(" ");
        os.append(this.originalSpan.get(i).get(TextAnnotation.class));
      }
      removed = os.toString();
    }
    if(posComma==-1 && posWH != -1 && this.headIndex < posWH){
      StringBuilder os = new StringBuilder();
      for(int i = 0; i < posWH-this.startIndex; i++){
        if(i > 0) os.append(" ");
        os.append(this.originalSpan.get(i).get(TextAnnotation.class));
      }
      removed = os.toString();
    }
    if(posComma==-1 && posWH == -1){
      removed = this.spanToString();
    }
    return removed;
  }

  public String longestNNPEndsWithHead (){
    String ret = "";
    for (int i = headIndex; i >=startIndex ; i--){
      String pos = sentenceWords.get(i).get(PartOfSpeechAnnotation.class);
      if(!pos.startsWith("NNP")) break;
      if(!ret.equals("")) ret = " "+ret;
      ret = sentenceWords.get(i).get(TextAnnotation.class)+ret;
    }
    return ret;
  }

  public String lowestNPIncludesHead (){
    String ret = "";
    Tree head = this.contextParseTree.getLeaves().get(this.headIndex);
    Tree lowestNP = head;
    String s;
    while(true) {
      if(lowestNP==null) return ret;
      s = ((CoreLabel) lowestNP.label()).get(ValueAnnotation.class);
      if(s.equals("NP") || s.equals("ROOT")) break;
      lowestNP = lowestNP.ancestor(1, this.contextParseTree);
    }
    if (s.equals("ROOT")) lowestNP = head;
    for (Tree t : lowestNP.getLeaves()){
      if (!ret.equals("")) ret = ret + " ";
      ret = ret + ((CoreLabel) t.label()).get(TextAnnotation.class);
    }
    if(!this.spanToString().contains(ret)) return this.sentenceWords.get(this.headIndex).get(TextAnnotation.class);
    return ret;
  }
  
  private List<String> getContextHelper(List<? extends CoreLabel> words) {
    List<List<CoreLabel>> namedEntities = new ArrayList<List<CoreLabel>>();
    List<CoreLabel> ne = new ArrayList<CoreLabel>();
    String previousNEType = "";
    int previousNEIndex = -1;
    for (int i = 0; i < words.size(); i++) {
      CoreLabel word = words.get(i);
      CoreLabel prevWord = i==0 ? null : words.get(i-1);
      if (isNE(word, prevWord)) {
        if (!word.ner().equals(previousNEType) || previousNEIndex != i-1) {
          ne = new ArrayList<CoreLabel>();
          namedEntities.add(ne);
        }
        ne.add(word);
        previousNEType = word.ner();
        previousNEIndex = i;
      }
    }

    List<String> neStrings = new ArrayList<String>();
    HashSet<String> hs = new HashSet<String>();
    for (int i = 0; i < namedEntities.size(); i++) {
      String ne_str = StringUtils.joinWords(namedEntities.get(i), " ");
      hs.add(ne_str);
    }
    neStrings.addAll(hs);
    return neStrings;
  }
  
  public List<String> getContext() {
    return getContextHelper(sentenceWords);
  }

  public List<String> getPremodifierContext() {
    List<String> neStrings = new ArrayList<String>();
    for (List<IndexedWord> words : getPremodifiers()) {
      neStrings.addAll(getContextHelper(words));
    }
    return neStrings;
  }

  // named entity helper function
  // modified to get cases like "iPhone", "iPad", and "webOS"
  private boolean isNE(CoreLabel word, CoreLabel prevWord) {
    String nerType = word.ner();
    return !(nerType.equals("TIME") || 
        nerType.equals("NUMBER") || 
        nerType.equals("ORDINAL") || 
        nerType.equals("MONEY") || 
        nerType.equals("DATE") || 
        nerType.equals("PERCENT")) 
      && 
      ((!nerType.equals("O") && word.word().length() > 1) || 
       (prevWord != null && !StringUtils.isPunct(prevWord.word()) && !word.word().equals(word.word().toLowerCase()) && word.word().length() > 1 && StringUtils.isAlpha(word.word())) ||
       (!word.word().substring(1).equals(word.word().substring(1).toLowerCase()) && word.word().length() > 1 && StringUtils.isAlpha(word.word())));
  }

  public boolean isNE() {
    int sentIndex = headWord.get(IndexAnnotation.class) - 1;
    CoreLabel prevWord = sentIndex == 0 ? null : sentenceWords.get(sentIndex - 1);
    return isNE(headWord, prevWord);
  }

  //Returns filtered premodifiers (no determiners or numerals)
  public ArrayList<ArrayList<IndexedWord>> getPremodifiers(){
    
    ArrayList<ArrayList<IndexedWord>> premod = new ArrayList<ArrayList<IndexedWord>>();
    
    IndexedWord head = dependency.getNodeByIndexSafe(headWord.get(IndexAnnotation.class));   
    if(head == null) return premod;
    for(Pair<GrammaticalRelation,IndexedWord> child : dependency.childPairs(head)){
      String function = child.first().getShortName();
      if(child.second().index() < headWord.index()
          && !child.second.tag().equals("DT") && !child.second.tag().equals("WRB")
          && !function.endsWith("det") && !function.equals("num") 
          && !function.equals("rcmod") && !function.equals("infmod")
          && !function.equals("partmod") && !function.equals("punct")){
        ArrayList<IndexedWord> phrase = new ArrayList<IndexedWord>(dependency.descendants(child.second()));
        Collections.sort(phrase);
        premod.add(phrase); 
      }
    }
    return premod;
  }
  
  // Returns filtered postmodifiers (no relative, -ed or -ing clauses)
  public ArrayList<ArrayList<IndexedWord>> getPostmodifiers(){

    ArrayList<ArrayList<IndexedWord>> postmod = new ArrayList<ArrayList<IndexedWord>>();

    IndexedWord head = dependency.getNodeByIndexSafe(headWord.get(IndexAnnotation.class));   
    if(head == null) return postmod;    
    for(Pair<GrammaticalRelation,IndexedWord> child : dependency.childPairs(head)){
      String function = child.first().getShortName();
      if(child.second().index() > headWord.index() &&         
          !function.endsWith("det") && !function.equals("num") 
          && !function.equals("rcmod") && !function.equals("infmod")
          && !function.equals("partmod") && !function.equals("punct")
          && !(function.equals("possessive") && dependency.descendants(child.second()).size() == 1)){
        ArrayList<IndexedWord> phrase = new ArrayList<IndexedWord>(dependency.descendants(child.second()));
        Collections.sort(phrase);
        postmod.add(phrase); 
      }
    }
    return postmod;
  }

  public String getPattern(List<CoreLabel> pTokens){

    ArrayList<String> phrase_string = new ArrayList<String>();
    String ne = "";
    for(CoreLabel token : pTokens){
      if(token.index() == headWord.index()){
        phrase_string.add(token.lemma());
        ne = "";

      } else if( (token.lemma().equals("and") || StringUtils.isPunct(token.lemma()))
          && pTokens.size() > pTokens.indexOf(token)+1
          && pTokens.indexOf(token) > 0
          && pTokens.get(pTokens.indexOf(token)+1).ner().equals(pTokens.get(pTokens.indexOf(token)-1).ner())){     

      } else if(token.index() == headWord.index()-1
          && token.ner().equals(nerString)){
        phrase_string.add(token.lemma());
        ne = "";

      } else if(!token.ner().equals("O")){
        if(!token.ner().equals(ne)){
          ne = token.ner();
          phrase_string.add("<"+ne+">");
        }

      } else {
        phrase_string.add(token.lemma());
        ne = "";
      }
    }
    return StringUtils.join(phrase_string);
  }

  public String[] getSplitPattern(){

    ArrayList<ArrayList<IndexedWord>> premodifiers = getPremodifiers();

    String[] components = new String[4]; 

    components[0] = headWord.lemma();

    if(premodifiers.size() == 0){
      components[1] = headWord.lemma();
      components[2] = headWord.lemma();
    } else if(premodifiers.size() == 1){ 
      ArrayList<CoreLabel> premod = new ArrayList<CoreLabel>();    
      premod.addAll(premodifiers.get(premodifiers.size()-1));
      premod.add(headWord);
      components[1] = getPattern(premod);
      components[2] = getPattern(premod);
    } else {
      ArrayList<CoreLabel> premod1 = new ArrayList<CoreLabel>();    
      premod1.addAll(premodifiers.get(premodifiers.size()-1));
      premod1.add(headWord);
      components[1] = getPattern(premod1);

      ArrayList<CoreLabel> premod2 = new ArrayList<CoreLabel>();
      for(ArrayList<IndexedWord> premodifier : premodifiers){
        premod2.addAll(premodifier);
      }    
      premod2.add(headWord);
      components[2] = getPattern(premod2);
    }

    components[3] = getPattern();
    return components;    
  }

  //TODO: NE head can span more than one word
  public String getPattern(){

    ArrayList<CoreLabel> pattern = new ArrayList<CoreLabel>();    
    for(ArrayList<IndexedWord> premodifier : getPremodifiers()){
      pattern.addAll(premodifier);
    }
    pattern.add(headWord);
    for(ArrayList<IndexedWord> postmodifier : getPostmodifiers()){
      pattern.addAll(postmodifier);
    }
    return getPattern(pattern);
  }
  
  public static final Set<String> indef_DTs = new HashSet<String>(Arrays.asList(
      "more",  "much", "many", "some", "no", "how", "other", "another", "every", "any"));
  
  public boolean hasIndefDT(){
    if(indef_DTs.contains(originalSpan.get(0).lemma())) return true;
    return false;
  }
  
  public boolean isCoordinated(){
    IndexedWord head = dependency.getNodeByIndexSafe(headWord.get(IndexAnnotation.class));   
    if(head == null) return false;
    for(Pair<GrammaticalRelation,IndexedWord> child : dependency.childPairs(head)){
      if(child.first().getShortName().equals("cc")) return true;
    }
    return false;
  }

  public String stringWithoutArticle(String str) {
    String ret = (str==null)? this.spanToString() : str;
    if (ret.startsWith("a ") || ret.startsWith("A ")) return ret.substring(2);
    else if(ret.startsWith("an ") || ret.startsWith("An ")) return ret.substring(3);
    else if(ret.startsWith("the ") || ret.startsWith("The ")) return ret.substring(4);
    return ret;
  }

  public List<String> preprocessSearchTerm (){
    List<String> searchTerms = new ArrayList<String>();
    String[] terms = new String[5];

    terms[0] = this.stringWithoutArticle(this.removePhraseAfterHead());
    terms[1] = this.stringWithoutArticle(this.lowestNPIncludesHead());
    terms[2] = this.stringWithoutArticle(this.longestNNPEndsWithHead());
    terms[3] = this.headString;
    terms[4] = this.headWord.get(LemmaAnnotation.class);

    for (String term : terms){

      if(term.contains("\"")) term = term.replace("\"", "\\\"");
      if(term.contains("(")) term = term.replace("(","\\(");
      if(term.contains(")")) term = term.replace(")", "\\)");
      if(term.contains("!")) term = term.replace("!", "\\!");
      if(term.contains(":")) term = term.replace(":", "\\:");
      if(term.contains("+")) term = term.replace("+", "\\+");
      if(term.contains("-")) term = term.replace("-", "\\-");
      if(term.contains("~")) term = term.replace("~", "\\~");
      if(term.contains("*")) term = term.replace("*", "\\*");
      if(term.contains("[")) term = term.replace("[", "\\[");
      if(term.contains("]")) term = term.replace("]", "\\]");
      if(term.contains("^")) term = term.replace("^", "\\^");
      if(term.equals("")) continue;

      if(term.equals("") || searchTerms.contains(term)) continue;
      if(term.equals(terms[3]) && !terms[2].equals("")) continue;
      searchTerms.add(term);
    }
    return searchTerms;
  }
  public String buildQueryText(List<String> terms) {
    String query = "";
    for (String t : terms){
      query += t + " ";
    }
    return query.trim();
  }

  private static Pair<IndexedWord, String> findDependentVerb(Mention m) {
    Pair<IndexedWord, String> ret = new Pair<IndexedWord, String>();
    int headIndex = m.headIndex+1;
    try {
      IndexedWord w = m.dependency.getNodeByIndex(headIndex);
      if(w==null) return ret;
      while (true) {
        IndexedWord p = null;
        for(Pair<GrammaticalRelation,IndexedWord> parent : m.dependency.parentPairs(w)){
          if(ret.second()==null) {
            String relation = parent.first().getShortName();
            ret.setSecond(relation);
          }
          p = parent.second();
        }
        if(p==null || p.get(PartOfSpeechAnnotation.class).startsWith("V")) {
          ret.setFirst(p);
          break;
        }
        if(w==p) return ret;
        w = p;
      }
    } catch (Exception e) {
      return ret;
    }
    return ret;
  }
  public boolean insideIn(Mention m){
    if(this.sentNum==m.sentNum &&
        m.startIndex <= this.startIndex
        && this.endIndex <= m.endIndex) return true;
    else return false;
  }
  protected boolean moreRepresentativeThan(Mention m){
    if(m==null) return true;
    if(mentionType!=m.mentionType) {
      if((mentionType==MentionType.PROPER && m.mentionType!=MentionType.PROPER)
          || (mentionType==MentionType.NOMINAL && m.mentionType==MentionType.PRONOMINAL)) return true;
      else return false;
    } else {
      if(headIndex-startIndex > m.headIndex - m.startIndex) return true;
      else if (sentNum < m.sentNum || (sentNum==m.sentNum && headIndex < m.headIndex)) return true;
      else return false;
    }
  }

  public static String sentenceWordsToString(Mention m) {
    StringBuilder sb = new StringBuilder();
    for(CoreLabel c : m.sentenceWords) {
      sb.append(c.get(TextAnnotation.class)).append(" ");
    }
    return sb.toString();
  }
  public static String mentionInfo(Mention m) {
    StringBuilder sb = new StringBuilder();
    sb.append(m.spanToString()).append(" ==> isEvent: ").append(m.isEvent);
    sb.append(", Head:").append(m.headString);
    sb.append(", twinless:" ).append( m.twinless);
    sb.append(", isReport: ").append(m.isReport);
    sb.append(", isVerb: ").append(m.isVerb);
    sb.append(", isPassive: ").append(m.isPassive);
    sb.append(", isEnumeration: ").append(m.isEnumeration);
    sb.append(", originalDocID: ").append(m.originalDocID);
    sb.append("\n\tparentInEnumeration: ").append(m.parentInEnumeration);
    sb.append("\n\tGender:").append(m.gender.toString());
    sb.append(", Number:").append(m.number.toString());
    sb.append(", Animacy:").append(m.animacy.toString());
    sb.append(", Person:").append(m.person.toString());
    sb.append(", NER:" ).append(m.nerString);
    sb.append(", Type:").append(m.mentionType.toString());
    sb.append(", utter: ").append(m.headWord.get(UtteranceAnnotation.class));
    sb.append(", speakerID: ").append(m.headWord.get(SpeakerAnnotation.class));
    sb.append("\n\tsrlPredicate: ").append( m.srlPredicates);
    sb.append(", Roles: ").append(m.srlArgs.toString());
    sb.append("\n\ttime: ").append(m.time);
    sb.append("\n\tlocation: ").append(m.location);
    sb.append("\n\tSuperSense:").append(m.headWord.get(WordSenseAnnotation.class));
    sb.append(", isSubject: ").append(m.isSubject);
    sb.append(", isDirectObject: ").append(m.isDirectObject);
    sb.append(", isIndirectObject: ").append(m.isIndirectObject);
    sb.append("\n\tdependents: ").append(m.dependents);
    //    sb.append("\n\tsynsets: ");
    //    if(m.synsets==null) sb.append(" NULL");
    //    else sb.append(m.synsets.synsets.toString());
    //    sb.append("\n\targuments: ").append(m.arguments);
    sb.append("\n\tappositions: ").append(m.appositions);
    sb.append("\n\tpredicate nominatives: ").append(m.predicateNominatives);
    sb.append("\n");
    sb.append("\t").append("subj: ");
    if(m.subject==null) sb.append("null\n");
    else sb.append(m.subject.spanToString()).append("\n");

    sb.append("\t").append("dobj: ");
    if(m.directObject==null) sb.append("null\n");
    else sb.append(m.directObject.spanToString()).append("\n");

    sb.append("\t").append("iobj: ");
    if(m.indirectObject==null) sb.append("null\n");
    else sb.append(m.indirectObject.spanToString()).append("\n");

    sb.append("\t").append("predicate: ");
    if(m.predicate==null) sb.append("null\n");
    else sb.append(m.predicate.spanToString()).append("\n");

    sb.append("\t").append("possessor: ");
    if(m.possessor==null) sb.append("null\n");
    else sb.append(m.possessor.spanToString()).append("\n");

    sb.append("\n");
    return sb.toString();
  }
}
