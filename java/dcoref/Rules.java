package edu.stanford.nlp.jcoref.dcoref;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import edu.stanford.nlp.jcoref.RuleBasedJointCorefSystem;
import edu.stanford.nlp.jcoref.dcoref.Dictionaries.Animacy;
import edu.stanford.nlp.jcoref.dcoref.Dictionaries.Gender;
import edu.stanford.nlp.jcoref.dcoref.Dictionaries.MentionType;
import edu.stanford.nlp.jcoref.dcoref.Dictionaries.Number;
import edu.stanford.nlp.jcoref.dcoref.Dictionaries.Person;
import edu.stanford.nlp.jcoref.dcoref.Mention.Argument;
import edu.stanford.nlp.jcoref.dcoref.SieveCoreferenceSystem.Semantics;
import edu.stanford.nlp.jcoref.dcoref.sievepasses.JointArgumentMatch;
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreAnnotations.LemmaAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.NamedEntityTagAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.PartOfSpeechAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.SpeakerAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TextAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.UtteranceAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.WordSenseAnnotation;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.IndexedWord;
import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.stats.Counters;
import edu.stanford.nlp.stats.IntCounter;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.IntPair;
import edu.stanford.nlp.util.IntTuple;
import edu.stanford.nlp.util.Pair;
import edu.stanford.nlp.util.Sets;

/**
 * Rules for coref system (mention detection, entity coref, event coref)
 * The name of the method for mention detection starts with detection,
 * for entity coref starts with entity, and for event coref starts with event.
 * 
 * @author heeyoung
 */
public class Rules {

  private static final boolean DEBUG = true;
  private static final double MENTION_2ndORDER_SIMILARITY_THRES = 0.7;
  private static final double SENTENCE_1stORDER_SIMILARITY_THRES = 0.7;

  //
  // Rules for mention detection
  //

  // TODO: move detection rule here


  //
  //  Rules for common coreference resolution
  //
  public static boolean enumerationIncompatible(CorefCluster menCluster, CorefCluster antCluster) {
    for(Mention m : menCluster.corefMentions) {
      if(!m.isEnumeration && m.parentInEnumeration==null) continue;
      for(Mention a : antCluster.corefMentions) {
        if(!a.isEnumeration && a.parentInEnumeration==null) continue;
        if(a.parentInEnumeration == m || m.parentInEnumeration == a
            || (m.parentInEnumeration!=null && m.parentInEnumeration == a.parentInEnumeration)) return true;
      }
    }
    return false;
  }

  //
  // Rules for entity coreference resolution
  //

  public static boolean entityPersonDisagree(Document document, CorefCluster mentionCluster, CorefCluster potentialAntecedent, Dictionaries dict){
    boolean disagree = false;
    for(Mention m : mentionCluster.getCorefMentions()) {
      for(Mention ant : potentialAntecedent.getCorefMentions()) {
        if(entityPersonDisagree(document, m, ant, dict)) {
          disagree = true;
        }
      }
    }
    if(disagree) return true;
    else return false;
  }
  /** Word inclusion except stop words  */
  public static boolean entityWordsIncluded(CorefCluster mentionCluster, CorefCluster potentialAntecedent, Mention mention, Mention ant) {
    Set<String> wordsExceptStopWords = new HashSet<String>(mentionCluster.words);
    wordsExceptStopWords.removeAll(Arrays.asList(new String[]{ "the","this", "mr.", "miss", "mrs.", "dr.", "ms.", "inc.", "ltd.", "corp.", "'s"}));
    wordsExceptStopWords.remove(mention.headString);
    if(potentialAntecedent.words.containsAll(wordsExceptStopWords)) return true;
    else return false;
  }

  /** Compatible modifier only  */
  public static boolean entityHaveIncompatibleModifier(CorefCluster mentionCluster, CorefCluster potentialAntecedent) {
    for(Mention m : mentionCluster.corefMentions){
      for(Mention ant : potentialAntecedent.corefMentions){
        if(m.haveIncompatibleModifier(ant)) return true;
      }
    }
    return false;
  }
  public static boolean entityIsRoleAppositive(CorefCluster mentionCluster, CorefCluster potentialAntecedent, Mention m1, Mention m2, Dictionaries dict) {
    if(!entityAttributesAgree(mentionCluster, potentialAntecedent)) return false;
    return m1.isRoleAppositive(m2, dict) || m2.isRoleAppositive(m1, dict);
  }
  public static boolean entityIsRelativePronoun(Mention m1, Mention m2) {
    return m1.isRelativePronoun(m2) || m2.isRelativePronoun(m1);
  }

  public static boolean entityIsAcronym(CorefCluster mentionCluster, CorefCluster potentialAntecedent) {
    for(Mention m : mentionCluster.corefMentions){
      if(m.isPronominal()) continue;
      for(Mention ant : potentialAntecedent.corefMentions){
        if(m.isAcronym(ant) || ant.isAcronym(m)) return true;
      }
    }
    return false;
  }

  public static boolean entityIsPredicateNominatives(CorefCluster mentionCluster, CorefCluster potentialAntecedent, Mention m1, Mention m2) {
    if(!entityAttributesAgree(mentionCluster, potentialAntecedent)) return false;
    if ((m1.startIndex <= m2.startIndex && m1.endIndex >= m2.endIndex)
        || (m1.startIndex >= m2.startIndex && m1.endIndex <= m2.endIndex)) {
      return false;
    }
    return m1.isPredicateNominatives(m2) || m2.isPredicateNominatives(m1);
  }

  public static boolean entityIsApposition(CorefCluster mentionCluster, CorefCluster potentialAntecedent, Mention m1, Mention m2) {
    if(!entityAttributesAgree(mentionCluster, potentialAntecedent)) return false;
    if(m1.mentionType==MentionType.PROPER && m2.mentionType==MentionType.PROPER) return false;
    if(m1.nerString.equals("LOCATION")) return false;
    return m1.isApposition(m2) || m2.isApposition(m1);
  }

  public static boolean entityNumberAgree(CorefCluster mentionCluster, CorefCluster antCluster){
    Set<Number> mNumbers = new HashSet<Number>(mentionCluster.numbers);
    mNumbers.retainAll(antCluster.numbers);
    if(mNumbers.size()==0 && !mentionCluster.numbers.contains(Number.UNKNOWN) && !antCluster.numbers.contains(Number.UNKNOWN)) return false;
    else return true;
  }
  public static boolean entityGenderAgree(CorefCluster mentionCluster, CorefCluster antCluster){
    Set<Gender> mGenders = new HashSet<Gender>(mentionCluster.genders);
    mGenders.retainAll(antCluster.genders);
    if(mGenders.size()==0 && !mentionCluster.genders.contains(Gender.UNKNOWN) && !antCluster.genders.contains(Gender.UNKNOWN)) return false;
    else return true;
  }
  public static boolean entityAnimacyAgree(CorefCluster mentionCluster, CorefCluster antCluster) {
    Set<Animacy> mAnimacies = new HashSet<Animacy>(mentionCluster.animacies);
    mAnimacies.retainAll(antCluster.animacies);
    if(mAnimacies.size()==0 && !mentionCluster.animacies.contains(Animacy.UNKNOWN) && !antCluster.animacies.contains(Animacy.UNKNOWN)) return false;
    else return true;
  }
  public static boolean entityNamedEntityTypeAgree(CorefCluster mentionCluster, CorefCluster antCluster){
    Set<String> mNERStrings = new HashSet<String>(mentionCluster.nerStrings);
    mNERStrings.retainAll(antCluster.nerStrings);
    if(mNERStrings.size()==0 && !mentionCluster.nerStrings.contains("O") && !mentionCluster.nerStrings.contains("MISC")
        && !antCluster.nerStrings.contains("O") && !antCluster.nerStrings.contains("MISC")) {
      return false;
    } else {
      return true;
    }
  }
  public static boolean entityNamedEntityTypeAgree(Mention m1, Mention m2) {
    if(m1.nerString.equals(m2.nerString) && !m1.nerString.equals("O")) return true;
    else return false;
  }
  public static boolean entityAttributesAgree(CorefCluster mentionCluster, CorefCluster antCluster){
    boolean numberAgree = entityNumberAgree(mentionCluster, antCluster);
    boolean genderAgree = entityGenderAgree(mentionCluster, antCluster);
    boolean animacyAgree = entityAnimacyAgree(mentionCluster, antCluster);
    boolean netypeAgree = entityNamedEntityTypeAgree(mentionCluster, antCluster);

    return (numberAgree && genderAgree && animacyAgree && netypeAgree);
  }
  public static boolean entityAttributesAgree_OLD(CorefCluster mentionCluster, CorefCluster potentialAntecedent){

    boolean hasExtraAnt = false;
    boolean hasExtraThis = false;

    // number
    if(!mentionCluster.numbers.contains(Number.UNKNOWN)){
      for(Number n : potentialAntecedent.numbers){
        if(n!=Number.UNKNOWN && !mentionCluster.numbers.contains(n)) hasExtraAnt = true;
      }
    }
    if(!potentialAntecedent.numbers.contains(Number.UNKNOWN)){
      for(Number n : mentionCluster.numbers){
        if(n!=Number.UNKNOWN && !potentialAntecedent.numbers.contains(n)) hasExtraThis = true;
      }
    }

    if(hasExtraAnt && hasExtraThis) return false;

    // gender
    hasExtraAnt = false;
    hasExtraThis = false;

    if(!mentionCluster.genders.contains(Gender.UNKNOWN)){
      for(Gender g : potentialAntecedent.genders){
        if(g!=Gender.UNKNOWN && !mentionCluster.genders.contains(g)) hasExtraAnt = true;
      }
    }
    if(!potentialAntecedent.genders.contains(Gender.UNKNOWN)){
      for(Gender g : mentionCluster.genders){
        if(g!=Gender.UNKNOWN && !potentialAntecedent.genders.contains(g)) hasExtraThis = true;
      }
    }
    if(hasExtraAnt && hasExtraThis) return false;

    // animacy
    hasExtraAnt = false;
    hasExtraThis = false;

    if(!mentionCluster.animacies.contains(Animacy.UNKNOWN)){
      for(Animacy a : potentialAntecedent.animacies){
        if(a!=Animacy.UNKNOWN && !mentionCluster.animacies.contains(a)) hasExtraAnt = true;
      }
    }
    if(!potentialAntecedent.animacies.contains(Animacy.UNKNOWN)){
      for(Animacy a : mentionCluster.animacies){
        if(a!=Animacy.UNKNOWN && !potentialAntecedent.animacies.contains(a)) hasExtraThis = true;
      }
    }
    if(hasExtraAnt && hasExtraThis) return false;

    // NE type
    hasExtraAnt = false;
    hasExtraThis = false;

    if(!mentionCluster.nerStrings.contains("O") && !mentionCluster.nerStrings.contains("MISC")){
      for(String ne : potentialAntecedent.nerStrings){
        if(!ne.equals("O") && !ne.equals("MISC") && !mentionCluster.nerStrings.contains(ne)) hasExtraAnt = true;
      }
    }
    if(!potentialAntecedent.nerStrings.contains("O") && !potentialAntecedent.nerStrings.contains("MISC")){
      for(String ne : mentionCluster.nerStrings){
        if(!ne.equals("O") && !ne.equals("MISC") && !potentialAntecedent.nerStrings.contains(ne)) hasExtraThis = true;
      }
    }
    return ! (hasExtraAnt && hasExtraThis);
  }

  public static boolean entityRelaxedHeadsAgreeBetweenMentions(CorefCluster mentionCluster, CorefCluster potentialAntecedent, Mention m, Mention ant) {
    if(m.isPronominal() || ant.isPronominal()) return false;
    if(m.headsAgree(ant)) return true;
    return false;
  }

  public static boolean entityHeadsAgree(CorefCluster mentionCluster, CorefCluster potentialAntecedent, Mention m, Mention ant, Dictionaries dict) {
    boolean headAgree = false;
    if(m.isPronominal() || ant.isPronominal()
        || dict.allPronouns.contains(m.lowercaseSpan)
        || dict.allPronouns.contains(ant.lowercaseSpan)) return false;
    for(Mention a : potentialAntecedent.corefMentions){
      if(a.headString.equals(m.headString)) headAgree= true;
    }
    return headAgree;
  }
  public static boolean entityExactStringMatch(Mention m, Mention a, Set<Mention> roleSet) {
    if(roleSet!=null && (roleSet.contains(m) || roleSet.contains(a))) return false;

    String mSpan = m.lowercaseSpan;
    String antSpan = a.lowercaseSpan;

    if(m.isPronominal() || a.isPronominal()) return false;
    if(mSpan.equals(antSpan)) return true;
    if(mSpan.equals(antSpan+" 's") || antSpan.equals(mSpan+" 's")
        || mSpan.equals(antSpan+" '") || antSpan.equals(mSpan+" '")) return true;
    return false;
  }

  public static boolean entityExactStringMatch(CorefCluster mentionCluster, CorefCluster potentialAntecedent, Dictionaries dict, Set<Mention> roleSet){
    for(Mention m : mentionCluster.corefMentions) {
      if(roleSet.contains(m)) return false;
    }
    for(Mention m : mentionCluster.corefMentions){
      for(Mention ant : potentialAntecedent.corefMentions){
        String mSpan = m.lowercaseSpan;
        String antSpan = ant.lowercaseSpan;

        if(m.isPronominal() || ant.isPronominal()
            || dict.allPronouns.contains(mSpan)
            || dict.allPronouns.contains(antSpan)) continue;
        if(mSpan.equals(antSpan)) return true;
        if(mSpan.equals(antSpan+" 's") || antSpan.equals(mSpan+" 's")
            || mSpan.equals(antSpan+" '") || antSpan.equals(mSpan+" '")) return true;
      }
    }
    return false;
  }

  /**
   * Exact string match except phrase after head (only for proper noun):
   * For dealing with a error like "[Mr. Bickford] <- [Mr. Bickford , an 18-year mediation veteran]"
   */
  public static boolean entityRelaxedExactStringMatch(
      CorefCluster mentionCluster,
      CorefCluster potentialAntecedent,
      Mention mention,
      Mention ant,
      Dictionaries dict,
      Set<Mention> roleSet){
    if(roleSet.contains(mention)) return false;
    if(mention.isPronominal() || ant.isPronominal()
        || dict.allPronouns.contains(mention.lowercaseSpan)
        || dict.allPronouns.contains(ant.lowercaseSpan)) return false;
    String mentionSpan = mention.removePhraseAfterHead().toLowerCase();
    String antSpan = ant.removePhraseAfterHead().toLowerCase();
    if(mentionSpan.equals("") || antSpan.equals("")) return false;

    if(mentionSpan.equals(antSpan) || mentionSpan.equals(antSpan+" 's") || antSpan.equals(mentionSpan+" 's")
        || mentionSpan.equals(antSpan+" '") || antSpan.equals(mentionSpan+" '")){
      return true;
    }
    return false;
  }
  public static boolean entityRelaxedExactStringMatch(
      Mention mention,
      Mention ant,
      Set<Mention> roleSet){
    if(roleSet.contains(mention)) return false;
    if(mention.isPronominal() || ant.isPronominal()) return false;
    String mentionSpan = mention.removePhraseAfterHead().toLowerCase();
    String antSpan = ant.removePhraseAfterHead().toLowerCase();
    if(mentionSpan.equals("") || antSpan.equals("")) return false;

    if(mentionSpan.equals(antSpan) || mentionSpan.equals(antSpan+" 's") || antSpan.equals(mentionSpan+" 's")
        || mentionSpan.equals(antSpan+" '") || antSpan.equals(mentionSpan+" '")){
      return true;
    }
    return false;
  }

  public static boolean entityBothHaveProper(CorefCluster mentionCluster,
      CorefCluster potentialAntecedent) {
    boolean mentionClusterHaveProper = false;
    boolean potentialAntecedentHaveProper = false;

    for (Mention m : mentionCluster.corefMentions) {
      if (m.mentionType==MentionType.PROPER) {
        mentionClusterHaveProper = true;
      }
    }
    for (Mention a : potentialAntecedent.corefMentions) {
      if (a.mentionType==MentionType.PROPER) {
        potentialAntecedentHaveProper = true;
      }
    }
    return (mentionClusterHaveProper && potentialAntecedentHaveProper);
  }
  public static boolean entitySameProperHeadLastWord(CorefCluster mentionCluster,
      CorefCluster potentialAntecedent, Mention mention, Mention ant) {
    for (Mention m : mentionCluster.getCorefMentions()){
      for (Mention a : potentialAntecedent.getCorefMentions()) {
        if (entitySameProperHeadLastWord(m, a)) return true;
      }
    }
    return false;
  }

  public static boolean entityAlias(CorefCluster mentionCluster, CorefCluster potentialAntecedent,
      Semantics semantics, Dictionaries dict) throws IOException {

    Mention mention = mentionCluster.getRepresentativeMention();
    Mention antecedent = potentialAntecedent.getRepresentativeMention();
    if(mention.mentionType!=MentionType.PROPER
        || antecedent.mentionType!=MentionType.PROPER) return false;

    if(semantics.wordnet.alias(mention, antecedent)) {
      return true;
    }
    return false;
  }
  public static boolean entityIWithinI(CorefCluster mentionCluster,
      CorefCluster potentialAntecedent, Dictionaries dict) {
    for(Mention m : mentionCluster.getCorefMentions()) {
      for(Mention a : potentialAntecedent.getCorefMentions()) {
        if(entityIWithinI(m, a, dict)) return true;
      }
    }
    return false;
  }

  /** Check whether two mentions are in i-within-i relation (Chomsky, 1981) */
  public static boolean entityIWithinI(Mention m1, Mention m2, Dictionaries dict){
    // check for nesting: i-within-i
    if(!m1.isApposition(m2) && !m2.isApposition(m1)
        && !m1.isRelativePronoun(m2) && !m2.isRelativePronoun(m1)
        && !m1.isRoleAppositive(m2, dict) && !m2.isRoleAppositive(m1, dict)
    ){
      if(m1.includedIn(m2) || m2.includedIn(m1)){
        return true;
      }
    }
    return false;
  }

  public static boolean entityIsSpeaker(Document document,
      Mention mention, Mention ant, Dictionaries dict) {
    if(document.speakerPairs.contains(new Pair<Integer, Integer>(mention.mentionID, ant.mentionID))
        || document.speakerPairs.contains(new Pair<Integer, Integer>(ant.mentionID, mention.mentionID))) {
      return true;
    }
    if(mention.headWord.get(SpeakerAnnotation.class).startsWith("PER")
        || ant.headWord.get(SpeakerAnnotation.class).startsWith("PER")) return false;

    if(mention.headWord.containsKey(SpeakerAnnotation.class)){
      for(String s : mention.headWord.get(SpeakerAnnotation.class).split(" ")) {
        if(ant.headString.equalsIgnoreCase(s)) return true;
      }
    }
    if(ant.headWord.containsKey(SpeakerAnnotation.class)){
      for(String s : ant.headWord.get(SpeakerAnnotation.class).split(" ")) {
        if(mention.headString.equalsIgnoreCase(s)) return true;
      }
    }
    return false;
  }

  public static boolean entityPersonDisagree(Document document, Mention m, Mention ant, Dictionaries dict) {
    boolean sameSpeaker = entitySameSpeaker(document, m, ant);

    if(sameSpeaker && m.person!=ant.person) {
      if((m.person==Person.IT && ant.person==Person.THEY) || (m.person==Person.THEY && ant.person==Person.IT)
          || (m.person==Person.THEY && ant.person==Person.THEY)) return false;
      else if(m.person!=Person.UNKNOWN && ant.person!=Person.UNKNOWN) return true;
    }
    if(sameSpeaker) {
      if(!ant.isPronominal()) {
        if(m.person==Person.I || m.person==Person.WE || m.person==Person.YOU) return true;
      } else if(!m.isPronominal()) {
        if(ant.person==Person.I || ant.person==Person.WE || ant.person==Person.YOU) return true;
      }
    }
    if(m.person==Person.YOU && ant.appearEarlierThan(m)) {
      int mUtter = m.headWord.get(UtteranceAnnotation.class);
      if(document.speakers.containsKey(mUtter-1)) {
        String previousSpeaker = document.speakers.get(mUtter-1);
        int previousSpeakerID;
        try {
          previousSpeakerID = Integer.parseInt(previousSpeaker);
        } catch (Exception e) {
          return true;
        }
        if(ant.corefClusterID!=document.allPredictedMentions.get(previousSpeakerID).corefClusterID
            && ant.person!=Person.I) return true;
      } else return true;
    } else if (ant.person==Person.YOU && m.appearEarlierThan(ant)) {
      int aUtter = ant.headWord.get(UtteranceAnnotation.class);
      if(document.speakers.containsKey(aUtter-1)) {
        String previousSpeaker = document.speakers.get(aUtter-1);
        int previousSpeakerID;
        try {
          previousSpeakerID = Integer.parseInt(previousSpeaker);
        } catch (Exception e) {
          return true;
        }
        if(m.corefClusterID!=document.allPredictedMentions.get(previousSpeakerID).corefClusterID
            && m.person!=Person.I) return true;
      } else return true;
    }
    return false;
  }

  public static boolean entitySameSpeaker(Document document, Mention m, Mention ant) {
    if(!m.headWord.containsKey(SpeakerAnnotation.class) ||
        !ant.headWord.containsKey(SpeakerAnnotation.class)){
      return false;
    }

    if(m.speakerID==-1 || ant.speakerID==-1)
      return (m.headWord.get(SpeakerAnnotation.class).equals(ant.headWord.get(SpeakerAnnotation.class)));

    int mSpeakerClusterID = document.allPredictedMentions.get(m.speakerID).corefClusterID;
    int antSpeakerClusterID = document.allPredictedMentions.get(ant.speakerID).corefClusterID;
    return (mSpeakerClusterID==antSpeakerClusterID);
  }

  public static boolean entitySubjectObject(Mention m1, Mention m2) {
    if(m1.sentNum != m2.sentNum) return false;
    if(m1.dependingVerb==null || m2.dependingVerb ==null) return false;
    if(m1.dependingVerb==m2.dependingVerb
        && ((m1.isSubject && (m2.isDirectObject || m2.isIndirectObject || m2.isPrepositionObject))
            || (m2.isSubject && (m1.isDirectObject || m1.isIndirectObject || m1.isPrepositionObject)))) return true;
    return false;
  }

  /** Check whether there is a new number in later mention */
  public static boolean entityNumberInLaterMention(Mention mention, Mention ant) {
    Set<String> antecedentWords = new HashSet<String>();
    Set<String> numbers = new HashSet<String>(Arrays.asList(new String[]{"one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "hundred", "thousand", "million", "billion"}));
    for (CoreLabel w : ant.originalSpan){
      antecedentWords.add(w.get(TextAnnotation.class));
    }
    for(CoreLabel w : mention.originalSpan) {
      String word = w.get(TextAnnotation.class);
      if(isNumeric(word) && !antecedentWords.contains(word)) return true;
      if(numbers.contains(word.toLowerCase()) && !antecedentWords.contains(word)) return true;
    }
    return false;
  }
  public static boolean isNumeric(String str){
    return str.matches("-?\\d+(.\\d+)?");
  }
  public static boolean entityHaveDifferentNumberMention(Mention m1, Mention m2) {
    Set<String> m1Number = new HashSet<String>();
    Set<String> m2Number = new HashSet<String>();
    Set<String> numbers = new HashSet<String>(Arrays.asList(new String[]{"one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "hundred", "thousand", "million", "billion"}));
    for (CoreLabel w : m1.originalSpan){
      String word = w.get(TextAnnotation.class).toLowerCase();
      if(isNumeric(word) || numbers.contains(word)) m1Number.add(word);
    }
    for(CoreLabel w : m2.originalSpan) {
      String word = w.get(TextAnnotation.class).toLowerCase();
      if(isNumeric(word) || numbers.contains(word)) m2Number.add(word);
    }
    Set<String> intersect = new HashSet<String>();
    intersect.addAll(m1Number);
    intersect.retainAll(m2Number);
    m1Number.removeAll(intersect);
    m2Number.removeAll(intersect);
    if(m1Number.size() > 0 && m2Number.size() > 0) return true;
    return false;
  }

  /** Have extra proper noun except strings involved in semantic match */
  public static boolean entityHaveExtraProperNoun(Mention m, Mention a, Set<String> exceptWords) {
    Set<String> mProper = new HashSet<String>();
    Set<String> aProper = new HashSet<String>();
    String mString = m.spanToString();
    String aString = a.spanToString();

    for (CoreLabel w : m.originalSpan){
      if (w.get(PartOfSpeechAnnotation.class).startsWith("NNP")) {
        mProper.add(w.get(TextAnnotation.class));
      }
    }
    for (CoreLabel w : a.originalSpan){
      if (w.get(PartOfSpeechAnnotation.class).startsWith("NNP")) {
        aProper.add(w.get(TextAnnotation.class));
      }
    }
    boolean mHasExtra = false;
    boolean aHasExtra = false;


    for (String s : mProper) {
      if (!aString.contains(s) && !exceptWords.contains(s.toLowerCase())) mHasExtra = true;
    }
    for (String s : aProper) {
      if (!mString.contains(s) && !exceptWords.contains(s.toLowerCase())) aHasExtra = true;
    }

    if(mHasExtra && aHasExtra) {
      return true;
    }
    return false;
  }

  /** Check whether two mentions have different locations */
  public static boolean entityHaveDifferentLocation(Mention m, Mention a, Dictionaries dict) {

    // state and country cannot be coref
    if((dict.statesAbbreviation.containsKey(a.spanToString()) || dict.statesAbbreviation.containsValue(a.spanToString()))
        && (m.headString.equalsIgnoreCase("country") || m.headString.equalsIgnoreCase("nation"))) return true;

    Set<String> locationM = new HashSet<String>();
    Set<String> locationA = new HashSet<String>();
    String mString = m.lowercaseSpan;
    String aString = a.lowercaseSpan;
    Set<String> locationModifier = new HashSet<String>(Arrays.asList("east", "west", "north", "south",
        "eastern", "western", "northern", "southern", "northwestern", "southwestern", "northeastern",
        "southeastern", "upper", "lower"));

    for (CoreLabel w : m.originalSpan){
      if (locationModifier.contains(w.get(TextAnnotation.class).toLowerCase())) return true;
      if (w.get(NamedEntityTagAnnotation.class).equals("LOCATION")) {
        String loc = w.get(TextAnnotation.class);
        if(dict.statesAbbreviation.containsKey(loc)) loc = dict.statesAbbreviation.get(loc);
        locationM.add(loc);
      }
    }
    for (CoreLabel w : a.originalSpan){
      if (locationModifier.contains(w.get(TextAnnotation.class).toLowerCase())) return true;
      if (w.get(NamedEntityTagAnnotation.class).equals("LOCATION")) {
        String loc = w.get(TextAnnotation.class);
        if(dict.statesAbbreviation.containsKey(loc)) loc = dict.statesAbbreviation.get(loc);
        locationA.add(loc);
      }
    }
    boolean mHasExtra = false;
    boolean aHasExtra = false;
    for (String s : locationM) {
      if (!aString.contains(s.toLowerCase())) mHasExtra = true;
    }
    for (String s : locationA) {
      if (!mString.contains(s.toLowerCase())) aHasExtra = true;
    }
    if(mHasExtra && aHasExtra) {
      return true;
    }
    return false;
  }
  
  /** Check whether two mentions have the same proper head words */
  public static boolean entitySameProperHeadLastWord(Mention m, Mention a) {
    if(!m.headString.equalsIgnoreCase(a.headString)
        || !m.sentenceWords.get(m.headIndex).get(PartOfSpeechAnnotation.class).startsWith("NNP")
        || !a.sentenceWords.get(a.headIndex).get(PartOfSpeechAnnotation.class).startsWith("NNP")) {
      return false;
    }
    if(!m.removePhraseAfterHead().toLowerCase().endsWith(m.headString)
        || !a.removePhraseAfterHead().toLowerCase().endsWith(a.headString)) {
      return false;
    }
    Set<String> mProperNouns = new HashSet<String>();
    Set<String> aProperNouns = new HashSet<String>();
    for (CoreLabel w : m.sentenceWords.subList(m.startIndex, m.headIndex)){
      if (w.get(PartOfSpeechAnnotation.class).startsWith("NNP")) {
        mProperNouns.add(w.get(TextAnnotation.class));
      }
    }
    for (CoreLabel w : a.sentenceWords.subList(a.startIndex, a.headIndex)){
      if (w.get(PartOfSpeechAnnotation.class).startsWith("NNP")) {
        aProperNouns.add(w.get(TextAnnotation.class));
      }
    }
    boolean mHasExtra = false;
    boolean aHasExtra = false;
    for (String s : mProperNouns) {
      if (!aProperNouns.contains(s)) mHasExtra = true;
    }
    for (String s : aProperNouns) {
      if (!mProperNouns.contains(s)) aHasExtra = true;
    }
    if(mHasExtra && aHasExtra) return false;
    return true;
  }

  /** true if two mentions are sharing the headword, but having different mention spans */
  public static boolean entityHeadSharing(Mention m, Mention a){
    if(m.headWord==a.headWord) {
      if(m.endIndex < a.endIndex && a.spanToString().contains(" and ")) {
        String nextPOS = m.sentenceWords.get(m.endIndex).get(PartOfSpeechAnnotation.class);
        if(nextPOS.equals(",") || nextPOS.equals("CC")) return false;
      }
      if(a.endIndex < m.endIndex && m.spanToString().contains(" and ")) {
        String nextPOS = a.sentenceWords.get(a.endIndex).get(PartOfSpeechAnnotation.class);
        if(nextPOS.equals(",") || nextPOS.equals("CC")) return false;
      }
      return true;
    }
    return false;
  }

  public static boolean entityHaveCorefSrlPredicate(CorefCluster menCluster, CorefCluster antCluster) {

    for(String role : menCluster.srlPredicates.keySet()) {
      if(!antCluster.srlPredicates.containsKey(role)) continue;
      Set<Integer> mSrlPredicateClusterID = new HashSet<Integer>();
      Set<Integer> aSrlPredicateClusterID = new HashSet<Integer>();
      for(Mention mPredicate : menCluster.srlPredicates.get(role)) {
        mSrlPredicateClusterID.add(mPredicate.corefClusterID);
      }
      for(Mention aPredicate : antCluster.srlPredicates.get(role)) {
        aSrlPredicateClusterID.add(aPredicate.corefClusterID);
      }
      mSrlPredicateClusterID.retainAll(aSrlPredicateClusterID);
      if(mSrlPredicateClusterID.size() > 0) return true;
    }
    return false;
  }
  public static boolean entityHaveSimilarSrlPredicate(CorefCluster menCluster, CorefCluster antCluster, Document doc) {

    for(String role : menCluster.srlPredicates.keySet()) {
      if(!antCluster.srlPredicates.containsKey(role)) continue;

      for(Mention mPredicate : menCluster.srlPredicates.get(role)) {
        for(Mention aPredicate : antCluster.srlPredicates.get(role)) {
          if(mPredicate==aPredicate) continue;
          IntPair idPair = new IntPair(Math.min(mPredicate.mentionID, aPredicate.mentionID), Math.max(mPredicate.mentionID, aPredicate.mentionID));
          if(doc.mentionSimilarInWN.get(idPair)) {
            return true;
          }
        }
      }
    }
    return false;
  }
  public static boolean entitySameSrlPredicate(Mention m, Mention a) {
    for(Mention mPredicate : m.srlPredicates.keySet()) {
      for(Mention aPredicate : a.srlPredicates.keySet()) {
        if(mPredicate.corefClusterID==aPredicate.corefClusterID
            && m.srlPredicates.get(mPredicate).equals(a.srlPredicates.get(aPredicate))) {
          return true;
        }
      }
    }
    return false;
  }

  // TODO: very slow - for temporary experiment
  public static boolean entityHaveSameContentWord(CorefCluster menCluster, CorefCluster antCluster) {
    for(Mention m : menCluster.getCorefMentions()) {
      for(Mention a : antCluster.getCorefMentions()) {
        if(entityHaveSameContentWord(m, a)) return true;
      }
    }
    return false;
  }
  public static boolean entityHaveSameContentWord(Mention m, Mention a) {
    for(CoreLabel ml : m.originalSpan) {
      for(CoreLabel al : a.originalSpan) {
        String mPos = ml.get(PartOfSpeechAnnotation.class);
        if(!mPos.startsWith("N") && !mPos.equals("CD") && !mPos.startsWith("JJ")) continue;
        if(ml.get(TextAnnotation.class).equalsIgnoreCase(al.get(TextAnnotation.class))) {
          return true;
        }
      }
    }
    return false;
  }
  public static boolean entityThesaurusSimilar(CorefCluster mC, CorefCluster aC, double threshold, Document document){
    if(mC.clusterID == aC.clusterID) return true;
    if(!entityAttributesAgree(mC, aC)) return false;
    double entityClusterSimilarityScore = 0;
    int countPairs = 0;
    for(Mention m : mC.getCorefMentions()) {
      for(Mention a : aC.getCorefMentions()) {
        if(m.isPronominal() || a.isPronominal()) continue;
        double entitySimilarityScore = getEntityThesaurusSimilarityScore(m, a, document);
        entityClusterSimilarityScore += entitySimilarityScore;
        countPairs++;
        //        if(entityThesaurusSimilar(m, a, threshold, document)) return true;
      }
    }
    return (entityClusterSimilarityScore/countPairs > threshold);
    //    return false;
  }
  private static double getEntityThesaurusSimilarityScore(Mention m, Mention a, Document document) {
    IntPair idPair = (m.mentionID < a.mentionID)? new IntPair(m.mentionID, a.mentionID) : new IntPair(a.mentionID, m.mentionID);
    return document.mentionSimilarity.get(idPair);
  }
  public static boolean entityThesaurusSimilar(Mention m, Mention a, double threshold, Document document) {
    IntPair idPair = (m.mentionID < a.mentionID)? new IntPair(m.mentionID, a.mentionID) : new IntPair(a.mentionID, m.mentionID);
    if(document.mentionSimilarity.get(idPair) > threshold) {
      return true;
    }
    return false;
  }
  public static boolean entityThesaurusSimilarWithCorefVerb(CorefCluster menCluster, CorefCluster antCluster, double threshold, Document document) {
    if(menCluster.clusterID == antCluster.clusterID) return true;
    if(!entityAttributesAgree(menCluster, antCluster)) return false;
    for(Mention m : menCluster.getCorefMentions()) {
      for(Mention a : antCluster.getCorefMentions()) {
        if(m.isPronominal() || a.isPronominal()) continue;
        if(entityThesaurusSimilar(m, a, threshold, document)
            && (entityHaveCorefLeftVerb(m, a, document) || entityHaveCorefRightVerb(m, a, document)
                || entitySameSrlPredicate(m, a))) {
          return true;
        }
      }
    }
    return false;
  }
  
 //COREF_DICT - strict cluster-cluster
 public static boolean entityClusterAllCorefDictionary(CorefCluster menCluster, CorefCluster antCluster, Dictionaries dict, int dictColumn, int freq){
   boolean ret = false;
   for(Mention men : menCluster.getCorefMentions()){
     if(men.isPronominal()) continue;
     for(Mention ant : antCluster.getCorefMentions()){   
       if(ant.isPronominal() || men.headWord.lemma().equals(ant.headWord.lemma())) continue;
       if(entityCorefDictionary(men, ant, dict, dictColumn, freq)){
         ret = true;
       } else {
         return false;
       }
     }
   }
   return ret; 
 }
  
  //COREF_DICT - mention-mention
  public static boolean entityCorefDictionary(Mention men, Mention ant, Dictionaries dict, int dictColumn, int freq){  
    
    Pair<String, String> mention_pair = new Pair<String, String>(men.getSplitPattern()[dictColumn-1].toLowerCase(), ant.getSplitPattern()[dictColumn-1].toLowerCase());     
    Pair<String, String> reversed_pair = new Pair<String, String>(ant.getSplitPattern()[dictColumn-1].toLowerCase(), men.getSplitPattern()[dictColumn-1].toLowerCase());
    
    int high_freq = 100;
    if(dictColumn == 1){ 
      high_freq = 75;
    } else if(dictColumn == 2){
      high_freq = 16;
    } else if(dictColumn == 3){
      high_freq = 16;
    } else if(dictColumn == 4){
      high_freq = 16;
    }

    if(dict.corefDictionary.get(dictColumn-1).getCount(mention_pair) > high_freq || dict.corefDictionary.get(dictColumn-1).getCount(reversed_pair) > high_freq) return true;

    if(dict.corefDictionary.get(dictColumn-1).getCount(mention_pair) > freq || dict.corefDictionary.get(dictColumn-1).getCount(reversed_pair) > freq){
        if(dict.corefDictionaryNPMI.getCount(mention_pair) > 0.18 || dict.corefDictionaryNPMI.getCount(reversed_pair) > 0.18) return true;
        if(!dict.corefDictionaryNPMI.containsKey(mention_pair) && !dict.corefDictionaryNPMI.containsKey(reversed_pair)) return true;
    }     
    return false; 
  }
  
  // Return true if the two mentions are less than n mentions apart in the same sent
  public static boolean entityTokenDistance(Mention men, Mention ant) {
    if(ant.sentNum == men.sentNum 
        && men.startIndex - ant.startIndex < 6) return true;
    return false;
  }
  
  public static boolean entityNumberAnimacyNEAgree(CorefCluster mentionCluster, CorefCluster antCluster) {   
    boolean numberAgree = entityNumberAgree(mentionCluster, antCluster);
    boolean animacyAgree = entityAnimacyAgree(mentionCluster, antCluster);
    boolean netypeAgree = entityNamedEntityTypeAgree(mentionCluster, antCluster);

    return (numberAgree && animacyAgree && netypeAgree);
  }

  public static boolean contextIncompatible(Mention men, Mention ant, Dictionaries dict) {
    String antHead = ant.headWord.get(TextAnnotation.class);
    if (ant.isNE() && ant.sentNum != men.sentNum && !isContextOverlapping(ant,men) && dict.signatures.containsKey(antHead)) {
      IntCounter<String> ranks = Counters.toRankCounter(dict.signatures.get(antHead));
      List<String> context;
      if (!men.getPremodifierContext().isEmpty()) {
        context = men.getPremodifierContext();
      } else {
        context = men.getContext();
      }
      if (!context.isEmpty()) {
        int highestRank = 100000;
        for (String w: context) {
          if (ranks.containsKey(w) && ranks.getIntCount(w) < highestRank) {
            highestRank = ranks.getIntCount(w);
          }
          // check in the other direction
          if (dict.signatures.containsKey(w)) {
            IntCounter<String> reverseRanks = Counters.toRankCounter(dict.signatures.get(w));
            if (reverseRanks.containsKey(antHead) && reverseRanks.getIntCount(antHead) < highestRank) {
              highestRank = reverseRanks.getIntCount(antHead);
            }
          }
        }
        if (highestRank > 10) return true;
      }
    }
    return false;
  }

  public static boolean sentenceContextIncompatible(Mention men, Mention ant, Dictionaries dict) {
    if (!ant.isNE() && ant.sentNum != men.sentNum && !men.isNE() && !isContextOverlapping(ant,men)) {
      List<String> context1 = !ant.getPremodifierContext().isEmpty() ? ant.getPremodifierContext() : ant.getContext();
      List<String> context2 = !men.getPremodifierContext().isEmpty() ? men.getPremodifierContext() : men.getContext();
      if (!context1.isEmpty() && !context2.isEmpty()) {
        int highestRank = 100000;
        for (String w1: context1) {
          for (String w2: context2) {
            // check the forward direction
            if (dict.signatures.containsKey(w1)) {
              IntCounter<String> ranks = Counters.toRankCounter(dict.signatures.get(w1));
              if (ranks.containsKey(w2) && ranks.getIntCount(w2) < highestRank) {
                highestRank = ranks.getIntCount(w2);
              }
            }
            // check in the other direction
            if (dict.signatures.containsKey(w2)) {
              IntCounter<String> reverseRanks = Counters.toRankCounter(dict.signatures.get(w2));
              if (reverseRanks.containsKey(w1) && reverseRanks.getIntCount(w1) < highestRank) {
                highestRank = reverseRanks.getIntCount(w1);
              }
            }
          }
        }
        if (highestRank > 10) return true;
      }
    }
    return false;
  }

  private static boolean isContextOverlapping(Mention m1, Mention m2) {
    Set<String> context1 = new HashSet<String>();
    Set<String> context2 = new HashSet<String>();
    context1.addAll(m1.getContext());
    context2.addAll(m2.getContext());
    return Sets.intersects(context1, context2);
  }
 
  private static boolean entityHaveCorefRightVerb(Mention m, Mention a, Document document) {
    IntTuple mPos = document.positions.get(m);
    IntTuple aPos = document.positions.get(a);
    Mention mRight = null;
    Mention aRight = null;

    List<Mention> mSentMentions = document.predictedOrderedMentionsBySentence.get(mPos.get(0));
    for(int i = mPos.get(1)+1 ; i < mSentMentions.size() ; i++) {
      if(mSentMentions.get(i).isVerb) {
        mRight = mSentMentions.get(i);
        break;
      }
    }
    List<Mention> aSentMentions = document.predictedOrderedMentionsBySentence.get(aPos.get(0));
    for(int i = aPos.get(1)+1 ; i < aSentMentions.size() ; i++) {
      if(aSentMentions.get(i).isVerb) {
        aRight = aSentMentions.get(i);
        break;
      }
    }
    if(mRight!=null && aRight!=null && mRight.corefClusterID == aRight.corefClusterID) {
      return true;
    }
    return false;
  }
  private static boolean entityHaveCorefLeftVerb(Mention m, Mention a, Document document) {
    IntTuple mPos = document.positions.get(m);
    IntTuple aPos = document.positions.get(a);
    Mention mLeft = null;
    Mention aLeft = null;

    List<Mention> mSentMentions = document.predictedOrderedMentionsBySentence.get(mPos.get(0));
    for(int i = mPos.get(1)-1 ; i >= 0 ; i--) {
      if(mSentMentions.get(i).isVerb) {
        mLeft = mSentMentions.get(i);
        break;
      }
    }
    List<Mention> aSentMentions = document.predictedOrderedMentionsBySentence.get(aPos.get(0));
    for(int i = aPos.get(1)-1 ; i >= 0 ; i--) {
      if(aSentMentions.get(i).isVerb) {
        aLeft = aSentMentions.get(i);
        break;
      }
    }
    if(mLeft!=null && aLeft!=null && mLeft.corefClusterID == aLeft.corefClusterID) {
      return true;
    }
    return false;
  }

  public static double entityArgMatchCount(CorefCluster menCluster, CorefCluster antCluster, Document document) {
    // check enumeration
    if(Rules.enumerationIncompatible(menCluster, antCluster)) return 0;

    double matchCount = 0;

    // role coreference (date, location, left, right mentions)
    for(String role : menCluster.srlRoles.keySet()) {
      if(antCluster.srlRoles.containsKey(role)) {
        Set<Integer> mArgClusterIDs = new HashSet<Integer>();
        Set<Integer> aArgClusterIDs = new HashSet<Integer>();
        for(Mention m : menCluster.srlRoles.get(role)) {
          if(m == null) continue;
          mArgClusterIDs.add(m.corefClusterID);
        }
        for(Mention a : antCluster.srlRoles.get(role)) {
          if(a == null) continue;
          aArgClusterIDs.add(a.corefClusterID);
        }
        mArgClusterIDs.retainAll(aArgClusterIDs);
        if(mArgClusterIDs.size() > 0) {
          matchCount++;
        }
      }
    }

    // predicate coreference
    for(String role : menCluster.srlPredicates.keySet()) {
      if(antCluster.srlPredicates.containsKey(role)) {
        Set<Integer> mPredicateClusterIDs = new HashSet<Integer>();
        Set<Integer> aPredicateClusterIDs = new HashSet<Integer>();
        for(Mention m : menCluster.srlPredicates.get(role)) {
          mPredicateClusterIDs.add(m.corefClusterID);
        }
        for(Mention a : antCluster.srlPredicates.get(role)) {
          aPredicateClusterIDs.add(a.corefClusterID);
        }
        mPredicateClusterIDs.retainAll(aPredicateClusterIDs);
        if(mPredicateClusterIDs.size() > 0) {
          matchCount += 2;
        }
      }
    }

    // arguments of predicate coreference
    // TODO

    // head agree
    Set<String> heads = new HashSet<String>(menCluster.heads);
    heads.retainAll(antCluster.heads);
    if(heads.size() > 0) {
      matchCount += 2;
    }

    // attributes agree
    //    if(entityAttributesAgree(menCluster, antCluster)) {
    //      matchCount++;
    //      matched.add("entityMultipleArgCoref: attributes agree");
    //    } else {
    //      matched.add("entityMultipleArgCoref: attributes notagree");
    //    }

    if(entityNumberAgree(menCluster, antCluster)) {
      matchCount++;
    }
    if(entityGenderAgree(menCluster, antCluster)) {
      matchCount++;
    }
    if(entityAnimacyAgree(menCluster, antCluster)) {
      matchCount++;
    }
    if(entityNamedEntityTypeAgree(menCluster, antCluster)) {
      matchCount++;
    }

    // mention 2nd order similarity
    int count = 0;
    double mentionSimilarity = 0.0;
    for(Mention m : menCluster.getCorefMentions()){
      for(Mention a : antCluster.getCorefMentions()) {
        if(m.isPronominal() || a.isPronominal()) continue;
        IntPair menPair = new IntPair(Math.min(m.mentionID, a.mentionID), Math.max(m.mentionID, a.mentionID));
        mentionSimilarity += document.mentionSimilarity.get(menPair);
        count++;
      }
    }
    if(mentionSimilarity/count > MENTION_2ndORDER_SIMILARITY_THRES) {
      matchCount += 2;
    }

    // sentence similarity
    boolean sentenceSimilar = false;
    for(Mention m : menCluster.getCorefMentions()){
      for(Mention a : antCluster.getCorefMentions()) {
        IntPair sentPair = new IntPair(Math.min(m.sentNum, a.sentNum), Math.max(m.sentNum, a.sentNum));
        if(m.sentNum != a.sentNum && document.sent1stOrderSimilarity.get(sentPair) > SENTENCE_1stORDER_SIMILARITY_THRES) {
          sentenceSimilar = true;
          break;
        }
      }
      if(sentenceSimilar) break;
    }
    if(sentenceSimilar) {
      matchCount++;
    }

    return matchCount;
  }
  public static double entityArgMatchCount(Mention m1, Mention m2, Document document) {
    // check enumeration
    if(m1.parentInEnumeration == m2 || m2.parentInEnumeration == m1) return 0;

    double matchCount = 0;

    // role coreference (date, location, left, right mentions)
    for(String role : m1.srlArgs.keySet()) {
      if(m1.srlArgs.get(role) != null && m2.srlArgs.containsKey(role) && m2.srlArgs.get(role)!=null) {
        if(m1.srlArgs.get(role).corefClusterID == m2.srlArgs.get(role).corefClusterID) matchCount++;
      }
    }

    // predicate coreference
    for(Mention pred1 : m1.srlPredicates.keySet()) {
      for(Mention pred2 : m2.srlPredicates.keySet()) {
        if(pred1.corefClusterID == pred2.corefClusterID 
            && m1.srlPredicates.get(pred1).equals(m2.srlPredicates.get(pred2))) matchCount += 2;
      }
    }

    // head agree
    if(m1.headString.equals(m2.headString)) matchCount += 2;

    if(m1.numbersAgree(m2)) matchCount++;
    if(m1.gendersAgree(m2)) matchCount++;
    if(m1.animaciesAgree(m2)) matchCount++;
    if(entityNamedEntityTypeAgree(m1, m2)) matchCount++;

    // mention 2nd order similarity
    IntPair menPair = new IntPair(Math.min(m1.mentionID, m2.mentionID), Math.max(m1.mentionID, m2.mentionID));
    if(document.mentionSimilarity.get(menPair) > MENTION_2ndORDER_SIMILARITY_THRES) matchCount += 2;
    
    // sentence similarity
    IntPair sentPair = new IntPair(Math.min(m1.sentNum, m2.sentNum), Math.max(m1.sentNum, m2.sentNum));
    if(m1.sentNum != m2.sentNum && document.sent1stOrderSimilarity.get(sentPair) > SENTENCE_1stORDER_SIMILARITY_THRES) {
      matchCount++;
    }
    return matchCount;
  }
  public static boolean entityMultipleArgCoref(CorefCluster menCluster, CorefCluster antCluster, Document document, double argMatchThreshold) {
    if(Rules.enumerationIncompatible(menCluster, antCluster)) return false;
    // # of correct or incorrect links when merged
    int correctEntityLinksCount = 0;
    int correctEventLinksCount = 0;
    int incorrectEntityLinksCount = 0;
    int incorrectEventLinksCount = 0;
    for(Mention m : menCluster.getCorefMentions()) {
      for(Mention a : antCluster.getCorefMentions()) {
        if(!document.allGoldMentions.containsKey(m.mentionID) || !document.allGoldMentions.containsKey(a.mentionID)) continue;
        if(document.allGoldMentions.get(m.mentionID).goldCorefClusterID == document.allGoldMentions.get(a.mentionID).goldCorefClusterID) {
          if(m.isEvent && a.isEvent) correctEventLinksCount++;
          else if(!m.isEvent && !a.isEvent) correctEntityLinksCount++;
        } else {
          if(m.isEvent && a.isEvent) incorrectEventLinksCount++;
          else if (!m.isEvent && !a.isEvent) incorrectEntityLinksCount++;
          else {
            incorrectEntityLinksCount++;
            incorrectEventLinksCount++;
          }
        }
      }
    }
    Set<String> matched = new HashSet<String>();

    StringBuilder sb = new StringBuilder();
    int matchCount = 0;

    // role coreference (date, location, left, right mentions)
    for(String role : menCluster.srlRoles.keySet()) {
      if(antCluster.srlRoles.containsKey(role)) {
        Set<Integer> mArgClusterIDs = new HashSet<Integer>();
        Set<Integer> aArgClusterIDs = new HashSet<Integer>();
        for(Mention m : menCluster.srlRoles.get(role)) {
          if(m == null) continue;
          mArgClusterIDs.add(m.corefClusterID);
        }
        for(Mention a : antCluster.srlRoles.get(role)) {
          if(a == null) continue;
          aArgClusterIDs.add(a.corefClusterID);
        }
        mArgClusterIDs.retainAll(aArgClusterIDs);
        if(mArgClusterIDs.size() > 0) {
          matchCount++;
          matched.add("entityMultipleArgCoref: srlRoles: "+role);
          sb.append("\n\tentityMultipleArgCoref: "+role+" matched");
        }
      }
    }

    // predicate coreference
    for(String role : menCluster.srlPredicates.keySet()) {
      if(antCluster.srlPredicates.containsKey(role)) {
        Set<Integer> mPredicateClusterIDs = new HashSet<Integer>();
        Set<Integer> aPredicateClusterIDs = new HashSet<Integer>();
        for(Mention m : menCluster.srlPredicates.get(role)) {
          mPredicateClusterIDs.add(m.corefClusterID);
        }
        for(Mention a : antCluster.srlPredicates.get(role)) {
          aPredicateClusterIDs.add(a.corefClusterID);
        }
        mPredicateClusterIDs.retainAll(aPredicateClusterIDs);
        if(mPredicateClusterIDs.size() > 0) {
          matchCount += 2;
          matched.add("entityMultipleArgCoref: srlPredicates: "+role);
          sb.append("\n\tentityMultipleArgCoref: "+menCluster.srlPredicates.get(role)+" matched in "+role);
        }
      }
    }

    // arguments of predicate coreference
    // TODO

    // head agree
    Set<String> heads = new HashSet<String>(menCluster.heads);
    heads.retainAll(antCluster.heads);
    if(heads.size() > 0) {
      sb.append("\n\tentityMultipleArgCoref: head matched: "+heads);
      matchCount += 2;
      matched.add("entityMultipleArgCoref: head agree");
    } else {
      matched.add("entityMultipleArgCoref: head notagree");
    }

    // attributes agree
    //    if(entityAttributesAgree(menCluster, antCluster)) {
    //      matchCount++;
    //      matched.add("entityMultipleArgCoref: attributes agree");
    //    } else {
    //      matched.add("entityMultipleArgCoref: attributes notagree");
    //    }

    if(entityNumberAgree(menCluster, antCluster)) {
      matchCount++;
      matched.add("entityMultipleArgCoref: number agree");
    } else {
      matched.add("entityMultipleArgCoref: number notagree");
    }
    if(entityGenderAgree(menCluster, antCluster)) {
      matchCount++;
      matched.add("entityMultipleArgCoref: gender agree");
    } else {
      matched.add("entityMultipleArgCoref: gender notagree");
    }
    if(entityAnimacyAgree(menCluster, antCluster)) {
      matchCount++;
      matched.add("entityMultipleArgCoref: animacy agree");
    } else {
      matched.add("entityMultipleArgCoref: animacy notagree");
    }
    if(entityNamedEntityTypeAgree(menCluster, antCluster)) {
      matchCount++;
      matched.add("entityMultipleArgCoref: named entity type agree");
    } else {
      matched.add("entityMultipleArgCoref: named entity type notagree");
    }

    // mention 2nd order similarity
    int count = 0;
    double mentionSimilarity = 0.0;
    for(Mention m : menCluster.getCorefMentions()){
      for(Mention a : antCluster.getCorefMentions()) {
        if(m.isPronominal() || a.isPronominal()) continue;
        IntPair menPair = new IntPair(Math.min(m.mentionID, a.mentionID), Math.max(m.mentionID, a.mentionID));
        mentionSimilarity += document.mentionSimilarity.get(menPair);
        count++;
      }
    }
    if(mentionSimilarity/count > MENTION_2ndORDER_SIMILARITY_THRES) {
      sb.append("\n\tentityMultipleArgCoref: mention 2nd order similar");
      matchCount += 2;
      matched.add("entityMultipleArgCoref: 2nd order mention similarity");
    } else {
      matched.add("entityMultipleArgCoref: 2nd order mention not similar");
    }

    // sentence similarity
    boolean sentenceSimilar = false;
    for(Mention m : menCluster.getCorefMentions()){
      for(Mention a : antCluster.getCorefMentions()) {
        IntPair sentPair = new IntPair(Math.min(m.sentNum, a.sentNum), Math.max(m.sentNum, a.sentNum));
        if(m.sentNum != a.sentNum && document.sent1stOrderSimilarity.get(sentPair) > SENTENCE_1stORDER_SIMILARITY_THRES) {
          sentenceSimilar = true;
          break;
        }
      }
      if(sentenceSimilar) break;
    }
    if(sentenceSimilar) {
      matchCount++;
      matched.add("entityMultipleArgCoref: sentence similarity");
      sb.append("\n\tentityMultipleArgCoref: sent 1st order similar");
    } else {
      matched.add("entityMultipleArgCoref: sentence not similar");
    }

    if(matchCount > argMatchThreshold) {
      RuleBasedJointCorefSystem.logger.fine(sb.append("\n===============================\n").toString());
      for(String s : matched) {
        RuleBasedJointCorefSystem.correctEntityLinkFound.incrementCount(s, correctEntityLinksCount);
        RuleBasedJointCorefSystem.correctEventLinkFound.incrementCount(s, correctEventLinksCount);
        RuleBasedJointCorefSystem.incorrectEntityLinkFound.incrementCount(s, incorrectEntityLinksCount);
        RuleBasedJointCorefSystem.incorrectEventLinkFound.incrementCount(s, incorrectEventLinksCount);
      }
      RuleBasedJointCorefSystem.correctEntityLinksFoundForThreshold.incrementCount(argMatchThreshold, correctEntityLinksCount);
      RuleBasedJointCorefSystem.incorrectEntityLinksFoundForThreshold.incrementCount(argMatchThreshold, incorrectEntityLinksCount);
      RuleBasedJointCorefSystem.correctEventLinksFoundForThreshold.incrementCount(argMatchThreshold, correctEventLinksCount);
      RuleBasedJointCorefSystem.incorrectEventLinksFoundForThreshold.incrementCount(argMatchThreshold, incorrectEventLinksCount);
      return true;
    }
    for(String s : matched) {
      RuleBasedJointCorefSystem.correctEntityLinkMissed.incrementCount(s, correctEntityLinksCount);
      RuleBasedJointCorefSystem.correctEventLinkMissed.incrementCount(s, correctEventLinksCount);
      RuleBasedJointCorefSystem.incorrectEntityLinkMissed.incrementCount(s, incorrectEntityLinksCount);
      RuleBasedJointCorefSystem.incorrectEventLinkMissed.incrementCount(s, incorrectEventLinksCount);
    }
    return false;
  }
  //
  // Rules for event coreference resolution
  //

  public static boolean eventHaveNotCorefArg(CorefCluster mentionCluster, CorefCluster potentialAntecedent, Mention mention2, Mention ant, Document document, WordNet wordnet, Dictionaries dict) {
    for(Mention m : mentionCluster.getCorefMentions()) {
      if(m.headWord.get(LemmaAnnotation.class).equals("be")) continue;
      for(Mention a : potentialAntecedent.getCorefMentions()) {
        if(a.headWord.get(LemmaAnnotation.class).equals("be")) continue;

        if(m.subject!=null && a.subject!=null && m.subject.corefClusterID!=a.subject.corefClusterID
            && WordNet.synonymInWN(m.subject, a.subject, wordnet, dict)) {
          return true;
        }
        if(m.directObject!=null && a.directObject!=null && m.directObject.corefClusterID!=a.directObject.corefClusterID
            && WordNet.synonymInWN(m.directObject, a.directObject, wordnet, dict)) {
          return true;
        }
      }
    }
    return false;
  }
  public static boolean eventLemmaObj(CorefCluster mentionCluster, CorefCluster potentialAntecedent, Mention mention2, Mention ant, Document document, Dictionaries dict) {
    for(Mention m : mentionCluster.getCorefMentions()) {
      for(Mention a : potentialAntecedent.getCorefMentions()) {
        if(eventLemmaObj(m, a, document, dict)) return true;
      }
    }
    return false;
  }

  public static boolean eventLemmaObj(Mention m, Mention a, Document document, Dictionaries dict) {
    if(a.synsets==null || m.synsets==null) return false;
    //    Synset[] nominalizedAnt = a.synsets.nominalizedSynsets;
    //    Synset[] nominalizedMen = m.synsets.nominalizedSynsets;

    for(String role : m.srlArgs.keySet()) {
      if(role.equals("A1") && m.srlArgs.get(role).headWord.get(LemmaAnnotation.class).equals(a.headWord.get(LemmaAnnotation.class))
          && dict.lightVerb.contains(m.headWord.get(LemmaAnnotation.class))) {
        return true;
      }
    }
    for(String role : a.srlArgs.keySet()) {
      if(role.equals("A1") && a.srlArgs.get(role).headWord.get(LemmaAnnotation.class).equals(m.headWord.get(LemmaAnnotation.class))
          && dict.lightVerb.contains(a.headWord.get(LemmaAnnotation.class))) {
        return true;
      }
    }
    return false;
  }
  public static boolean eventSameLemma(Mention m, Mention a) {
    if(m.headWord.get(LemmaAnnotation.class).equals(a.headWord.get(LemmaAnnotation.class))) return true;
    else return false;
  }

  public static boolean eventSimilarInWN(CorefCluster mC, CorefCluster aC, double threshold, Document document) {
    int total = 0;
    int similarCnt = 0;
    for(Mention m : mC.getCorefMentions()) {
      for(Mention a : aC.getCorefMentions()) {
        total++;
        IntPair idPair = new IntPair(Math.min(m.mentionID, a.mentionID), Math.max(m.mentionID, a.mentionID));
        if(document.mentionSimilarInWN.get(idPair)) {
          similarCnt++;
        }
      }
    }
    return ((similarCnt*1.0/total) >= threshold);
  }

  public static boolean eventShareArgument(Mention m, Mention a) {
    Set<String> intersection = new HashSet<String>(m.arguments.keySet());
    intersection.retainAll(a.arguments.keySet());
    for(String s : intersection) {
      Argument arg1 = m.arguments.get(s);
      Argument arg2 = a.arguments.get(s);
      if(arg1.word.get(TextAnnotation.class).equals(arg2.word.get(TextAnnotation.class))
          && arg1.relation!=null && arg2.relation!=null
          && arg1.relation.equals(arg2.relation)) {
        return true;
      }
    }
    return false;
  }

  public static boolean eventHaveCorefArgument(Mention m, Mention a, Document doc) {

    if(m.isPassive == a.isPassive) {
      if(m.subject!=null && a.subject!=null && m.subject.corefClusterID == a.subject.corefClusterID) return true;
      if(m.directObject!=null && a.directObject!=null && m.directObject.corefClusterID == a.directObject.corefClusterID) return true;
    } else {
      if(m.directObject!=null && a.subject!=null && m.directObject.corefClusterID == a.subject.corefClusterID) return true;
      if(m.subject!=null && a.directObject!=null && m.subject.corefClusterID == a.directObject.corefClusterID) return true;
    }

    if((m.possessor!=null && a.subject!=null && m.possessor.corefClusterID == a.subject.corefClusterID)
        || (m.subject!=null && a.possessor!=null && m.subject.corefClusterID==a.possessor.corefClusterID)) return true;

    if(m.indirectObject!=null && a.indirectObject!=null && m.indirectObject.corefClusterID == a.indirectObject.corefClusterID) return true;

    IntTuple mPos = doc.positions.get(m);
    IntTuple aPos = doc.positions.get(a);

    List<Mention> mSentMentions = doc.getOrderedMentions().get(mPos.get(0));
    List<Mention> aSentMentions = doc.getOrderedMentions().get(aPos.get(0));

    Mention mPre = (mPos.get(1) > 0)? mSentMentions.get(mPos.get(1)-1) : null;
    Mention aPre = (aPos.get(1) > 0)? aSentMentions.get(aPos.get(1)-1) : null;

    Mention mPost = (mPos.get(1) < mSentMentions.size()-1)? mSentMentions.get(mPos.get(1)+1) : null;
    Mention aPost = (aPos.get(1) < aSentMentions.size()-1)? aSentMentions.get(aPos.get(1)+1) : null;

    // if there are coreferent mentions before and after an event mention, return true;
    if(mPre!=null && aPre!=null) {
      if(mPost!=null && aPost!=null) {
        if(mPre.corefClusterID==aPre.corefClusterID && mPost.corefClusterID==aPost.corefClusterID) return true;
      } else {
        if(mPre.corefClusterID==aPre.corefClusterID) return true;
      }
    } else {
      if(mPost!=null && aPost!=null) {
        if(mPost.corefClusterID==aPost.corefClusterID) return true;
      }
    }

    return false;
  }
  public static boolean eventSameSchema(Mention m, Mention ant, Dictionaries dict) {
    String mLemma = m.headWord.get(LemmaAnnotation.class);
    String aLemma = ant.headWord.get(LemmaAnnotation.class);
    if(dict.schemaID.containsKey(mLemma) && dict.schemaID.containsKey(aLemma)
        && dict.schemaID.get(mLemma) == dict.schemaID.get(aLemma)) return true;
    return false;
  }
  public static boolean eventHaveCorefEntityInSentence(Mention mention2, Mention ant, Document document) {
    if(mention2.sentNum == ant.sentNum) return false;   // skip if they are in a same sentence.
    List<Mention> m = document.getOrderedMentions().get(mention2.sentNum);
    List<Mention> a = document.getOrderedMentions().get(ant.sentNum);

    Set<Integer> corefIDSet = new HashSet<Integer>();
    for(Mention men : m) {
      corefIDSet.add(men.corefClusterID);
    }
    for(Mention men : a) {
      if(corefIDSet.contains(men.corefClusterID)) {
        return true;
      }
    }
    return false;
  }

  public static boolean eventIWithinI(CorefCluster mC, CorefCluster aC, Document doc) {
    for(Mention m : mC.getCorefMentions()) {
      for(Mention a : aC.getCorefMentions()) {
        if(eventIWithinI(m, a, doc)) {
          if(mC.clusterID==4266 && aC.clusterID==4352) {
            System.err.println();
          }
          return true;
        }
      }
    }
    return false;
  }

  public static boolean eventIWithinI(Mention mention, Mention ant, Document doc) {
    if(mention.sentNum!=ant.sentNum) return false;

    Tree mTree = mention.mentionSubTree.parent(mention.contextParseTree);
    Tree aTree = ant.mentionSubTree.parent(ant.contextParseTree);

    if(mTree.dominates(ant.mentionSubTree) || aTree.dominates(mention.mentionSubTree)) {
      //      if(doc.allGoldMentions.containsKey(mention.mentionID) && doc.allGoldMentions.containsKey(ant.mentionID)) {
      //        if(doc.allGoldMentions.get(mention.mentionID).goldCorefClusterID == doc.allGoldMentions.get(ant.mentionID).goldCorefClusterID) {
      //          System.err.println();
      //        }
      //      }
      return true;
    }
    return false;
  }

  public static boolean eventHaveCorefSrlArgument(CorefCluster mC, CorefCluster aC){
    for(Mention m : mC.getCorefMentions()) {
      for(Mention a : aC.getCorefMentions()) {
        if(eventHaveCorefSrlArgument(m, a)) return true;
      }
    }
    return false;
  }

  public static boolean eventHaveCorefSrlArgument(Mention mention, Mention ant) {
    Map<String, Mention> mentionSrlArgs = mention.srlArgs;
    Map<String, Mention> antSrlArgs = ant.srlArgs;
    Set<String> roleCollapsed1 = new HashSet<String>(Arrays.asList("A2", "A3", "A4", "A5"));
    Set<String> roleCollapsed2 = new HashSet<String>(Arrays.asList("AM-DIR", "AM-LOC"));

    for(String mRole : mentionSrlArgs.keySet()) {
      for(String aRole : antSrlArgs.keySet()) {
        if(mRole.equals(aRole) || (roleCollapsed1.contains(mRole) && roleCollapsed1.contains(aRole)) || (roleCollapsed2.contains(mRole) && roleCollapsed2.contains(aRole))) {
          if(mentionSrlArgs.get(mRole).corefClusterID == antSrlArgs.get(aRole).corefClusterID) {
            return true;
          }
        }
      }
    }
    return false;
  }
  public static boolean eventHaveSameHeadSrlArgument(CorefCluster menCluster, CorefCluster antCluster) {
    for(Mention m : menCluster.getCorefMentions()) {
      for(Mention a : antCluster.getCorefMentions()) {
        if(eventHaveSameHeadSrlArgument(m, a)) return true;
      }
    }
    return false;
  }
  public static boolean eventHaveSameHeadSrlArgument(Mention mention, Mention ant) {
    Map<String, Mention> mentionSrlArgs = mention.srlArgs;
    Map<String, Mention> antSrlArgs = ant.srlArgs;
    Set<String> roleCollapsed1 = new HashSet<String>(Arrays.asList("A2", "A3", "A4", "A5"));
    Set<String> roleCollapsed2 = new HashSet<String>(Arrays.asList("AM-DIR", "AM-LOC"));

    for(String mRole : mentionSrlArgs.keySet()) {
      for(String aRole : antSrlArgs.keySet()) {
        if(mRole.equals(aRole) || (roleCollapsed1.contains(mRole) && roleCollapsed1.contains(aRole)) || (roleCollapsed2.contains(mRole) && roleCollapsed2.contains(aRole))) {
          if(mentionSrlArgs.get(mRole).headString.equals(antSrlArgs.get(aRole).headString)) {
            return true;
          }
        }
      }
    }
    return false;
  }
  public static boolean eventHavePossibleCorefSrlArgument(CorefCluster mC, CorefCluster aC, Dictionaries dict) {
    for(Mention m : mC.getCorefMentions()) {
      for(Mention a : aC.getCorefMentions()) {
        if(eventHavePossibleCorefSrlArgument(m, a, dict)) return true;
      }
    }
    return false;
  }

  public static boolean eventHavePossibleCorefSrlArgument(Mention mention, Mention ant, Dictionaries dict) {
    Map<String, Mention> mentionSrlArgs = mention.srlArgs;
    Map<String, Mention> antSrlArgs = ant.srlArgs;
    Set<String> roleCollapsed1 = new HashSet<String>(Arrays.asList("A2", "A3", "A4", "A5"));
    Set<String> roleCollapsed2 = new HashSet<String>(Arrays.asList("AM-DIR", "AM-LOC"));

    for(String mRole : mentionSrlArgs.keySet()) {
      for(String aRole : antSrlArgs.keySet()) {
        if(mRole.equals(aRole) || (roleCollapsed1.contains(mRole) && roleCollapsed1.contains(aRole)) || (roleCollapsed2.contains(mRole) && roleCollapsed2.contains(aRole))) {
          Mention mArg = mentionSrlArgs.get(mRole);
          Mention aArg = antSrlArgs.get(aRole);
          if(mArg.corefClusterID == aArg.corefClusterID) return true;
          if(!mArg.attributesAgree(aArg, dict)) return false;
          if(entityHaveSameContentWord(mArg, aArg)) {
            return true;
          }
        }
      }
    }
    return false;
  }
  public static boolean eventDekangLinSimpleMatch(Mention mention, Mention ant, Dictionaries dict) {
    String mLemma = mention.headWord.get(LemmaAnnotation.class);
    String aLemma = ant.headWord.get(LemmaAnnotation.class);
    if(dict.thesaurusVerb.containsKey(mLemma) && dict.thesaurusVerb.containsKey(aLemma)
        && dict.thesaurusVerb.get(mLemma).contains(aLemma)
        && dict.thesaurusVerb.get(aLemma).contains(mLemma)) {
      return true;
    }
    return false;
  }
  public static boolean eventSuperSenseMatch(Mention mention, Mention ant, Dictionaries dict) {
    String mSense = mention.headWord.get(WordSenseAnnotation.class);
    String aSense = ant.headWord.get(WordSenseAnnotation.class);
    if(mSense == null || aSense == null || mSense.equals("0") || aSense.equals("0")) return false;
    if(mSense.split("\\.")[1].equals(aSense.split("\\.")[1])) return true;
    return false;
  }
  public static boolean eventSentSimilar(CorefCluster mC, CorefCluster aC, Document document) {
    for(Mention m : mC.getCorefMentions()) {
      for(Mention a : aC.getCorefMentions()) {
        if (eventSentSimilar(m, a, document)) return true;
      }
    }
    return false;
  }
  public static boolean eventSentSimilar(Mention mention, Mention ant, Document document) {

    IntPair sentPair = new IntPair(Math.min(mention.sentNum, ant.sentNum), Math.max(mention.sentNum, ant.sentNum));

    double sim = (mention.sentNum==ant.sentNum)? 1 : document.sent2ndOrderSimilarity.get(sentPair);
    if(sim > 0.3 && sim < 1) return true;
    return false;
  }

  public static boolean eventThesaurusSimilarSrlArg(CorefCluster mC, CorefCluster aC, double threshold, Document document) {
    Map<String, Set<CorefCluster>> mSrlArgClusters = new HashMap<String, Set<CorefCluster>>();
    Map<String, Set<CorefCluster>> aSrlArgClusters = new HashMap<String, Set<CorefCluster>>();

    for(String role : mC.srlRoles.keySet()) {
      Set<CorefCluster> argCluster = new HashSet<CorefCluster>();
      mSrlArgClusters.put(role, argCluster);
      for(Mention arg : mC.srlRoles.get(role)) {
        argCluster.add(document.corefClusters.get(arg.corefClusterID));
      }
    }
    for(String role : aC.srlRoles.keySet()) {
      Set<CorefCluster> argCluster = new HashSet<CorefCluster>();
      aSrlArgClusters.put(role, argCluster);
      for(Mention arg : aC.srlRoles.get(role)) {
        argCluster.add(document.corefClusters.get(arg.corefClusterID));
      }
    }

    for(String role :  mSrlArgClusters.keySet()) {
      if(!aSrlArgClusters.containsKey(role)) continue;
      for(CorefCluster mArgCluster : mSrlArgClusters.get(role)) {
        for(CorefCluster aArgCluster : aSrlArgClusters.get(role)) {
          if(entityThesaurusSimilar(mArgCluster, aArgCluster, threshold, document)) {
            return true;
          }
        }
      }
    }

    // slow
    //    for(Mention m : mC.getCorefMentions()) {
    //      for(Mention a : aC.getCorefMentions()) {
    //        if(eventThesaurusSimilarSrlArg(m, a, threshold, document)) return true;
    //      }
    //    }
    return false;
  }

  private static boolean eventThesaurusSimilarSrlArg(Mention m, Mention a, double threshold, Document document) {
    Map<String, Mention> mentionSrlArgs = m.srlArgs;
    Map<String, Mention> antSrlArgs = a.srlArgs;
    Set<String> roleCollapsed1 = new HashSet<String>(Arrays.asList("A2", "A3", "A4", "A5"));
    Set<String> roleCollapsed2 = new HashSet<String>(Arrays.asList("AM-DIR", "AM-LOC"));

    for(String mRole : mentionSrlArgs.keySet()) {
      for(String aRole : antSrlArgs.keySet()) {
        if(mRole.equals(aRole) || (roleCollapsed1.contains(mRole) && roleCollapsed1.contains(aRole)) || (roleCollapsed2.contains(mRole) && roleCollapsed2.contains(aRole))) {
          Mention mArg = mentionSrlArgs.get(mRole);
          Mention aArg = antSrlArgs.get(aRole);
          if(mArg.corefClusterID == aArg.corefClusterID) return true;
          if(entityThesaurusSimilar(document.corefClusters.get(mArg.corefClusterID), document.corefClusters.get(aArg.corefClusterID), threshold, document)) {
            return true;
          }
        }
      }
    }
    return false;
  }
  public static boolean eventHaveNotCorefSrlArgument(CorefCluster mC, CorefCluster aC) {
    for(Mention m : mC.getCorefMentions()) {
      for(Mention a : aC.getCorefMentions()) {
        if(eventHaveNotCorefSrlArgument(m, a)) return true;
      }
    }
    return false;
  }
  private static boolean eventHaveNotCorefSrlArgument(Mention m, Mention a) {
    Map<String, Mention> mentionSrlArgs = m.srlArgs;
    Map<String, Mention> antSrlArgs = a.srlArgs;

    for(String role : mentionSrlArgs.keySet()) {
      if(antSrlArgs.containsKey(role)
          && mentionSrlArgs.get(role).corefClusterID != antSrlArgs.get(role).corefClusterID) {
        return true;
      }
    }
    return false;
  }
  public static boolean eventFrequentVerb(Mention m, Dictionaries dict, double idf_cutoff) {
    Double score = dict.tfIdf.idfScore.get(m.headWord.get(LemmaAnnotation.class));
    if(score != null && score < idf_cutoff) return true;
    else return false;
  }
  public static boolean eventHaveCorefLeftMention(Mention mention, Mention ant, Document document) {
    IntTuple mPos = document.positions.get(mention);
    IntTuple aPos = document.positions.get(ant);

    if(mPos.get(1) == 0 || aPos.get(1) == 0) return false;
    Mention mLeft = document.predictedOrderedMentionsBySentence.get(mPos.get(0)).get(mPos.get(1)-1);
    Mention aLeft = document.predictedOrderedMentionsBySentence.get(aPos.get(0)).get(aPos.get(1)-1);

    if(mLeft.corefClusterID == aLeft.corefClusterID) return true;
    return false;
  }
  public static boolean eventHaveCorefRightMention(Mention mention, Mention ant, Document document) {
    IntTuple mPos = document.positions.get(mention);
    IntTuple aPos = document.positions.get(ant);

    if(mPos.get(1) == document.predictedOrderedMentionsBySentence.get(mPos.get(0)).size()-1
        || aPos.get(1) == document.predictedOrderedMentionsBySentence.get(aPos.get(0)).size()-1) return false;
    Mention mRight = document.predictedOrderedMentionsBySentence.get(mPos.get(0)).get(mPos.get(1)+1);
    Mention aRight = document.predictedOrderedMentionsBySentence.get(aPos.get(0)).get(aPos.get(1)+1);

    if(mRight.corefClusterID == aRight.corefClusterID) return true;
    return false;
  }
  public static boolean eventWNSimilaritySurfaceContext(CorefCluster menCluster, CorefCluster antCluster, WordNet wordnet, Document document, Dictionaries dict) {
    for(Mention m : menCluster.getCorefMentions()) {
      for(Mention a : antCluster.getCorefMentions()) {
        boolean similar = WordNet.similarInWN(m, a, wordnet, dict);
        boolean left = eventHaveCorefLeftMention(m, a, document);
        boolean right = eventHaveCorefRightMention(m, a, document);
        if(similar && left && right) {
          return true;
        }
      }
    }
    return false;
  }
  public static boolean eventSrlArgumentMatch(CorefCluster menCluster, CorefCluster antCluster, Document document) {
    for(Mention m : menCluster.getCorefMentions()) {
      for(Mention a : antCluster.getCorefMentions()) {
        if(m.sentNum==a.sentNum) return false;
        // A0 && A1 matching
        if(m.srlArgs.containsKey("A0") && m.srlArgs.containsKey("A1")
            && a.srlArgs.containsKey("A0") && a.srlArgs.containsKey("A1")) {
          if(m.srlArgs.get("A0").corefClusterID == a.srlArgs.get("A0").corefClusterID
              && m.srlArgs.get("A1").corefClusterID == a.srlArgs.get("A1").corefClusterID) {
            return true;
          }
        }

        // left && right mention matching
        //        boolean left = eventHaveCorefLeftMention(m, a, document);
        //        boolean right = eventHaveCorefRightMention(m, a, document);
        //        if(left && right) {
        //          return true;
        //        }
      }
    }
    return false;
  }
  public static boolean eventHaveCommonSrlArg(CorefCluster menCluster, CorefCluster antCluster) {
    Set<String> mRoles = new HashSet<String>(menCluster.srlRoles.keySet());
    mRoles.retainAll(antCluster.srlRoles.keySet());
    if(mRoles.size()==0) {
      return false;
    }
    else {
      return true;
    }
  }
  public static boolean eventClusterSynonym(CorefCluster mC, CorefCluster aC, double threshold, Document document) {
    int total = 0;
    int synonymCnt = 0;
    for(Mention m : mC.getCorefMentions()) {
      for(Mention a : aC.getCorefMentions()) {
        total++;
        IntPair idPair = new IntPair(Math.min(m.mentionID, a.mentionID), Math.max(m.mentionID, a.mentionID));
        if(document.mentionSynonymInWN.contains(idPair)) {
          synonymCnt++;
        }
      }
    }
    return ((synonymCnt*1.0/total) >= threshold);
  }
  public static boolean eventClusterSimilar(CorefCluster mC, CorefCluster aC, double threshold, Document document) {
    int total = 0;
    int similarCnt = 0;
    for(Mention m : mC.getCorefMentions()) {
      for(Mention a : aC.getCorefMentions()) {
        total++;
        IntPair idPair = new IntPair(Math.min(m.mentionID, a.mentionID), Math.max(m.mentionID, a.mentionID));
        if(document.mentionSimilarInWN.get(idPair)) {
          similarCnt++;
        }
      }
    }
    return ((similarCnt*1.0/total) >= threshold);
  }
  public static boolean eventSameLocation(Mention mention, Mention ant) {
    if(mention.location!=null && ant.location!=null
        && mention.location.corefClusterID != ant.location.corefClusterID) return false;
    return true;
  }
  public static boolean eventSameTime(Mention mention, Mention ant) {
    if(mention.time!=null && ant.time!=null
        && mention.time.corefClusterID != ant.time.corefClusterID) return false;
    return true;
  }

  public static double eventArgMatchCount(CorefCluster menCluster, CorefCluster antCluster, Document document){
    if(Rules.enumerationIncompatible(menCluster, antCluster)) return 0;
    double matchCount = 0;
    for(String role : menCluster.srlRoles.keySet()) {
      if(antCluster.srlRoles.containsKey(role)) {
        boolean roleCoref = false;
        for(Mention m : menCluster.srlRoles.get(role)) {
          for(Mention a : antCluster.srlRoles.get(role)) {
            if(m==null || a==null) continue;
            if(role.equals("AM-LOC") || role.equals("AM-TMP")) {
              if(m.corefClusterID==a.corefClusterID && m.sentNum != a.sentNum) roleCoref = true;
            } else {
              if(m.corefClusterID==a.corefClusterID) {
                roleCoref = true;
                break;
              }
            }
          }
        }
        if(roleCoref) {
          if(role.equals("RIGHT-MENTION") || role.equals("A1") || role.equals("A0")) matchCount += 2;
          else matchCount++;
        }
      }
    }
    boolean synonym = false;
    boolean sameLemma = false;
    for(Mention m : menCluster.getCorefMentions()){
      for(Mention a : antCluster.getCorefMentions()) {
        IntPair idPair = new IntPair(Math.min(m.mentionID, a.mentionID), Math.max(m.mentionID, a.mentionID));
        if(document.mentionSynonymInWN.contains(idPair)) {
          synonym = true;
        }
        if(m.headWord.get(LemmaAnnotation.class).equals(a.headWord.get(LemmaAnnotation.class))) {
          sameLemma = true;
        }
      }
      if(synonym && sameLemma) {
        break;
      }
    }
    if(synonym && !sameLemma) {
      matchCount += 2;
    }
    if(sameLemma) {
      matchCount += 5;
    }
    // sentence similarity
    boolean sentenceSimilar = false;
    for(Mention m : menCluster.getCorefMentions()){
      for(Mention a : antCluster.getCorefMentions()) {
        IntPair sentPair = new IntPair(Math.min(m.sentNum, a.sentNum), Math.max(m.sentNum, a.sentNum));
        if(m.sentNum != a.sentNum && document.sent1stOrderSimilarity.get(sentPair) > SENTENCE_1stORDER_SIMILARITY_THRES) {
          matchCount++;
          sentenceSimilar = true;
          break;
        }
      }
      if(sentenceSimilar) break;
    }
    return matchCount;
  }
  public static double eventArgMatchCount(Mention m1, Mention m2, Document document){
    if(m1.parentInEnumeration == m2 || m2.parentInEnumeration == m1) return 0;

    double matchCount = 0;
    
    for(String role : m1.srlArgs.keySet()) {
      if(m2.srlArgs.containsKey(role)) {
        if(m1.srlArgs.get(role)!=null && m2.srlArgs.get(role) != null && m1.srlArgs.get(role).corefClusterID == m2.srlArgs.get(role).corefClusterID){
          if((role.equals("AM-LOC") || role.equals("AM-TMP")) && m1.sentNum == m2.sentNum) continue;
          if(role.equals("RIGHT-MENTION") || role.equals("A1") || role.equals("A0")) matchCount += 2;
          else matchCount++;
        }
      }
    }
    
    boolean synonym = false;
    boolean sameLemma = false;
    
    IntPair idPair = new IntPair(Math.min(m1.mentionID, m2.mentionID), Math.max(m1.mentionID, m2.mentionID));
    if(document.mentionSynonymInWN.contains(idPair)) synonym = true;
    if(m1.headWord.get(LemmaAnnotation.class).equals(m2.headWord.get(LemmaAnnotation.class))) sameLemma = true;

    if(synonym && !sameLemma) {
      matchCount += 2;
    }
    if(sameLemma) {
      matchCount += 5;
    }
    
    IntPair sentPair = new IntPair(Math.min(m1.sentNum, m2.sentNum), Math.max(m1.sentNum, m2.sentNum));
    if(m1.sentNum != m2.sentNum && document.sent1stOrderSimilarity.get(sentPair) > SENTENCE_1stORDER_SIMILARITY_THRES) {
      matchCount++;
    }
    return matchCount;
  }
  public static boolean eventMultipleArgCoref(CorefCluster menCluster, CorefCluster antCluster, Document document, double argMatchThreshold) {
    if(Rules.enumerationIncompatible(menCluster, antCluster)) return false;

    // # of correct or incorrect links when merged
    int correctEntityLinksCount = 0;
    int correctEventLinksCount = 0;
    int incorrectEntityLinksCount = 0;
    int incorrectEventLinksCount = 0;
    for(Mention m : menCluster.getCorefMentions()) {
      for(Mention a : antCluster.getCorefMentions()) {
        if(!document.allGoldMentions.containsKey(m.mentionID) || !document.allGoldMentions.containsKey(a.mentionID)) continue;
        if(document.allGoldMentions.get(m.mentionID).goldCorefClusterID == document.allGoldMentions.get(a.mentionID).goldCorefClusterID) {
          if(m.isEvent && a.isEvent) correctEventLinksCount++;
          else if(!m.isEvent && !a.isEvent) correctEntityLinksCount++;
        } else {
          if(m.isEvent && a.isEvent) incorrectEventLinksCount++;
          else if (!m.isEvent && !a.isEvent) incorrectEntityLinksCount++;
          else {
            incorrectEntityLinksCount++;
            incorrectEventLinksCount++;
          }
        }
      }
    }
    Set<String> matched = new HashSet<String>();

    int matchCount = 0;
    StringBuilder sb = new StringBuilder();
    for(String role : menCluster.srlRoles.keySet()) {
      if(antCluster.srlRoles.containsKey(role)) {
        boolean roleCoref = false;
        for(Mention m : menCluster.srlRoles.get(role)) {
          for(Mention a : antCluster.srlRoles.get(role)) {
            if(m==null || a==null) continue;
            if(role.equals("AM-LOC") || role.equals("AM-TMP")) {
              if(m.corefClusterID==a.corefClusterID && m.sentNum != a.sentNum) roleCoref = true;
            } else {
              if(m.corefClusterID==a.corefClusterID) roleCoref = true;
            }
          }
        }
        if(roleCoref) {
          if(role.equals("RIGHT-MENTION") || role.equals("A1") || role.equals("A0")) matchCount += 2;
          else matchCount++;
          matched.add("eventMultipleArgCoref: srlRoles: "+role);
          sb.append("\n\teventMultipleArgCoref: "+role+" matched");
        }
      }
    }
    boolean synonym = false;
    boolean sameLemma = false;
    for(Mention m : menCluster.getCorefMentions()){
      for(Mention a : antCluster.getCorefMentions()) {
        IntPair idPair = new IntPair(Math.min(m.mentionID, a.mentionID), Math.max(m.mentionID, a.mentionID));
        if(document.mentionSynonymInWN.contains(idPair)) {
          synonym = true;
        }
        if(m.headWord.get(LemmaAnnotation.class).equals(a.headWord.get(LemmaAnnotation.class))) {
          sameLemma = true;
        }
      }
      if(synonym && sameLemma) {
        break;
      }
    }
    if(synonym && !sameLemma) {
      matchCount += 2;
      matched.add("eventMultipleArgCoref: synonym but not same lemma");
      sb.append("\n\teventMultipleArgCoref: synonym but not same lemma");
    }
    if(sameLemma) {
      matchCount += 2;
      matched.add("eventMultipleArgCoref: sameLemma");
      sb.append("\n\teventMultipleArgCoref: same lemma");
    }

    // sentence similarity
    boolean sentenceSimilar = false;
    for(Mention m : menCluster.getCorefMentions()){
      for(Mention a : antCluster.getCorefMentions()) {
        IntPair sentPair = new IntPair(Math.min(m.sentNum, a.sentNum), Math.max(m.sentNum, a.sentNum));
        if(m.sentNum != a.sentNum && document.sent1stOrderSimilarity.get(sentPair) > SENTENCE_1stORDER_SIMILARITY_THRES) {
          sb.append("\n\teventMultipleArgCoref: sentence 1st order similar");
          matchCount++;
          matched.add("eventMultipleArgCoref: sentence similarity");
          sentenceSimilar = true;
          break;
        }
      }
      if(sentenceSimilar) break;
    }

    // nominal event vs verb event
    //    if(CorefCluster.nominalCluster(menCluster) || CorefCluster.nominalCluster(antCluster)) {
    //      argMatchThreshold /= 3;
    //    }

    if(matchCount > argMatchThreshold) {
      RuleBasedJointCorefSystem.logger.fine(sb.append("\n===============================\n").toString());
      for(String s : matched) {
        RuleBasedJointCorefSystem.correctEntityLinkFound.incrementCount(s, correctEntityLinksCount);
        RuleBasedJointCorefSystem.correctEventLinkFound.incrementCount(s, correctEventLinksCount);
        RuleBasedJointCorefSystem.incorrectEntityLinkFound.incrementCount(s, incorrectEntityLinksCount);
        RuleBasedJointCorefSystem.incorrectEventLinkFound.incrementCount(s, incorrectEventLinksCount);
      }
      RuleBasedJointCorefSystem.correctEntityLinksFoundForThreshold.incrementCount(argMatchThreshold, correctEntityLinksCount);
      RuleBasedJointCorefSystem.incorrectEntityLinksFoundForThreshold.incrementCount(argMatchThreshold, incorrectEntityLinksCount);
      RuleBasedJointCorefSystem.correctEventLinksFoundForThreshold.incrementCount(argMatchThreshold, correctEventLinksCount);
      RuleBasedJointCorefSystem.incorrectEventLinksFoundForThreshold.incrementCount(argMatchThreshold, incorrectEventLinksCount);

      return true;
    }
    for(String s : matched) {
      RuleBasedJointCorefSystem.correctEntityLinkMissed.incrementCount(s, correctEntityLinksCount);
      RuleBasedJointCorefSystem.correctEventLinkMissed.incrementCount(s, correctEventLinksCount);
      RuleBasedJointCorefSystem.incorrectEntityLinkMissed.incrementCount(s, incorrectEntityLinksCount);
      RuleBasedJointCorefSystem.incorrectEventLinkMissed.incrementCount(s, incorrectEventLinksCount);
    }
    return false;

  }

  public static boolean eventSameLemma(CorefCluster menCluster,
      CorefCluster antCluster) {
    for(Mention m : menCluster.corefMentions) {
      for(Mention ant : antCluster.corefMentions) {
        if(m.isPronominal() || ant.isPronominal()) continue;
//        if(m.isVerb && ant.isVerb) continue;
        if(eventSameLemma(m, ant)) {
          return true;
        }
      }
    }
    return false;
  }

  public static boolean entityHaveSameStringPredicate(CorefCluster menCluster, CorefCluster antCluster, Document document) {
    JointArgumentMatch.calculateCentroid(menCluster, document, false);
    JointArgumentMatch.calculateCentroid(antCluster, document, false);
    for(String feature : menCluster.predictedCentroid.keySet()) {
      if(!feature.startsWith("SRLPREDS") || !antCluster.predictedCentroid.containsKey(feature)) continue;
      Counter<String> menFeature = menCluster.predictedCentroid.get(feature);
      Counter<String> antFeature = antCluster.predictedCentroid.get(feature);
      Set<String> menPreds = new HashSet<String>(menCluster.predictedCentroid.get(feature).keySet());
      Set<String> antPreds = new HashSet<String>(antCluster.predictedCentroid.get(feature).keySet());
      menPreds.retainAll(antPreds);
      if(menPreds.size() > 0) {
        return true;
      }
    }
    return false;
  }

  public static boolean eventHaveSameStringArgs(CorefCluster menCluster, CorefCluster antCluster, Document document) {
    JointArgumentMatch.calculateCentroid(menCluster, document, false);
    JointArgumentMatch.calculateCentroid(antCluster, document, false);
    for(String feature : menCluster.predictedCentroid.keySet()) {
      if(!feature.startsWith("SRLROLES") || !antCluster.predictedCentroid.containsKey(feature)) continue;
      Counter<String> menFeature = menCluster.predictedCentroid.get(feature);
      Counter<String> antFeature = antCluster.predictedCentroid.get(feature);
      Set<String> menPreds = new HashSet<String>(menCluster.predictedCentroid.get(feature).keySet());
      Set<String> antPreds = new HashSet<String>(antCluster.predictedCentroid.get(feature).keySet());
      menPreds.retainAll(antPreds);
      if(menPreds.size() > 0) {
        return true;
      }
    }
    return false;
  }
}
