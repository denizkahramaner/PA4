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

package edu.stanford.nlp.jcoref.dcoref.sievepasses;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.Set;

import edu.stanford.nlp.jcoref.RuleBasedJointCorefSystem;
import edu.stanford.nlp.jcoref.dcoref.Constants;
import edu.stanford.nlp.jcoref.dcoref.CorefCluster;
import edu.stanford.nlp.jcoref.dcoref.Dictionaries;
import edu.stanford.nlp.jcoref.dcoref.Dictionaries.Number;
import edu.stanford.nlp.jcoref.dcoref.Dictionaries.Person;
import edu.stanford.nlp.jcoref.dcoref.Document;
import edu.stanford.nlp.jcoref.dcoref.Document.DocType;
import edu.stanford.nlp.jcoref.dcoref.Mention;
import edu.stanford.nlp.jcoref.dcoref.Rules;
import edu.stanford.nlp.jcoref.dcoref.SieveCoreferenceSystem;
import edu.stanford.nlp.jcoref.dcoref.SieveCoreferenceSystem.Semantics;
import edu.stanford.nlp.jcoref.dcoref.SieveOptions;
import edu.stanford.nlp.jcoref.dcoref.WordNet;
import edu.stanford.nlp.ling.CoreAnnotations.PartOfSpeechAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.SpeakerAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.UtteranceAnnotation;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.util.IntPair;
import edu.stanford.nlp.util.Pair;

/**
 *  Base class for Coref Sieve
 *  Each sieve extends this class, and set flags for its own options in the constructor
 *
 */
public abstract class DeterministicCorefSieve  {
  public SieveOptions flags;
  private final Set<IntPair> alreadyComparedClusterPair;
  private final Map<IntPair, Boolean> enumerationCompared;

  /** Initialize flagSet */
  public DeterministicCorefSieve(){
    flags = new SieveOptions();
    alreadyComparedClusterPair = new HashSet<IntPair>();
    enumerationCompared = new HashMap<IntPair, Boolean>();
  }
  public void init(Properties props)
  {
  }

  public String flagsToString() { return flags.toString(); }

  public boolean useRoleSkip() { return flags.USE_ROLE_SKIP; }

  /** Skip this mention? (search pruning) */
  public boolean skipThisMention(Document document, Mention m1, CorefCluster c, Dictionaries dict) {
    boolean skip = false;

    // only do for the first mention in its cluster
    if(!flags.USE_EXACTSTRINGMATCH && !flags.USE_ROLEAPPOSITION && !flags.USE_PREDICATENOMINATIVES
        && !flags.USE_ACRONYM && !flags.USE_APPOSITION && !flags.USE_RELATIVEPRONOUN && !flags.DO_HEADSHARING
        //        && !flags.ORACLE_EVENT
        && !c.getFirstMention().equals(m1)) {
      return true;
    }

    if(Constants.USE_DISCOURSE_SALIENCE)  {
      SieveCoreferenceSystem.logger.finest("DOING COREF FOR:\t" + m1.spanToString());
      if(m1.appositions == null && m1.predicateNominatives == null
          && (m1.lowercaseSpan.startsWith("a ") || m1.lowercaseSpan.startsWith("an "))
          && !flags.USE_EXACTSTRINGMATCH)  {
        skip = true; // A noun phrase starting with an indefinite article - unlikely to have an antecedent (e.g. "A commission" was set up to .... )
      }
      if(dict.indefinitePronouns.contains(m1.lowercaseSpan))  {
        skip = true; // An indefinite pronoun - unlikely to have an antecedent (e.g. "Some" say that... )
      }
      for(String indef : dict.indefinitePronouns){
        if(m1.lowercaseSpan.startsWith(indef + " ")) {
          skip = true; // A noun phrase starting with an indefinite adjective - unlikely to have an antecedent (e.g. "Another opinion" on the topic is...)
          break;
        }
      }

      if(skip) {
        SieveCoreferenceSystem.logger.finest("MENTION SKIPPED:\t" + m1.spanToString() + "(" + m1.sentNum + ")"+"\toriginalRef: "+m1.originalRef + " in discourse "+m1.headWord.get(UtteranceAnnotation.class));
      }
    }

    return skip;
  }

  /**
   * Checks if two clusters are coreferent according to our sieve pass constraints
   * @param document
   */
  public boolean coreferent(Document document, CorefCluster menCluster,
      CorefCluster antCluster,
      Mention mention,
      Mention ant,
      Dictionaries dict,
      Set<Mention> roleSet,
      Semantics semantics) {
    IntPair clusterIDPair = new IntPair(Math.min(menCluster.getClusterID(), antCluster.getClusterID()), Math.max(menCluster.getClusterID(), antCluster.getClusterID()));
    if(flags.CLUSTER_MATCH) {
      if(alreadyComparedClusterPair.contains(clusterIDPair)) {
        return false;
      } else {
        alreadyComparedClusterPair.add(clusterIDPair);
      }
    }

    if(!enumerationCompared.containsKey(clusterIDPair)) {
      boolean incompatible = Rules.enumerationIncompatible(menCluster, antCluster);
      enumerationCompared.put(clusterIDPair, incompatible);
      if(incompatible) {
        document.errorLog.put(new IntPair(Math.min(mention.mentionID, ant.mentionID), Math.max(mention.mentionID, ant.mentionID)), "Enumeration Incompatible");
        return false;
      }
    } else {
      if(enumerationCompared.get(clusterIDPair)) {
        document.errorLog.put(new IntPair(Math.min(mention.mentionID, ant.mentionID), Math.max(mention.mentionID, ant.mentionID)), "Enumeration Incompatible");
        return false;
      }
    }

    // oracle matching for analysis
    Map<Integer, Mention> golds = document.allGoldMentions;
    if(golds.containsKey(mention.mentionID) && golds.containsKey(ant.mentionID)) {
      if((flags.ORACLE_ENTITY && !golds.get(mention.mentionID).isEvent && !golds.get(ant.mentionID).isEvent)
          || (flags.ORACLE_EVENT && golds.get(mention.mentionID).isEvent && golds.get(ant.mentionID).isEvent)
      ) {
        //      if(!mention2.isEvent && !ant.isEvent) {
        if(golds.containsKey(mention.mentionID) && golds.containsKey(ant.mentionID)) {
          if(golds.get(mention.mentionID).goldCorefClusterID == golds.get(ant.mentionID).goldCorefClusterID) {
            return true;
          }
          return false;
        }
      }
    }

    if(!flags.ALLOW_VERB_NOUN_MATCH && mention.isVerb!=ant.isVerb) return false; 
    if(flags.WITHIN_DOC && !mention.originalDocID.equals(ant.originalDocID)) return false;
       
    if(flags.FOR_EVENT) {
      return coreferentEvent(document, menCluster, antCluster, mention, ant, dict, semantics);
    } else {
      return coreferentEntity(document, menCluster, antCluster, mention, ant, dict, roleSet, semantics);
    }
  }

  private boolean coreferentEntity(Document document, CorefCluster menCluster, CorefCluster antCluster, Mention mention, Mention ant, Dictionaries dict, Set<Mention> roleSet, Semantics semantics) {
    if(mention.isVerb || ant.isVerb) return false;

    boolean ret = false;
    IntPair idPair = new IntPair(Math.min(mention.mentionID, ant.mentionID), Math.max(mention.mentionID, ant.mentionID));
    Mention menRep = menCluster.getRepresentativeMention();

    if(flags.DO_HEADSHARING && Rules.entityHeadSharing(mention, ant)) {
      document.errorLog.put(idPair, "headsharing");
      return true;
    }

    if(flags.DO_PRONOUN && Math.abs(mention.sentNum-ant.sentNum) > 3
        && mention.person!=Person.I && mention.person!=Person.YOU) {
      document.errorLog.put(idPair, "pronoun distance too far");
      return false;
    }
    if(mention.lowercaseSpan.equals("this") && Math.abs(mention.sentNum-ant.sentNum) > 3) {
      document.errorLog.put(idPair, "antecedent for 'this' is too far");
      return false;
    }
    if(mention.person==Person.YOU && document.docType==DocType.ARTICLE
        && mention.headWord.get(SpeakerAnnotation.class).equals("PER0")) {
      document.errorLog.put(idPair, "generic you in non-conversation");
      return false;
    }
    if(document.conllDoc != null) {
      if(ant.generic && ant.person==Person.YOU || mention.generic) {
        document.errorLog.put(idPair, "generic");
        return false;
      }
      if(mention.insideIn(ant) || ant.insideIn(mention)) {
        document.errorLog.put(idPair, "i-within-i");
        return false;
      }
    }

    if(flags.USE_DISCOURSEMATCH) {
      String mString = menRep.lowercaseSpan;
      String antString = ant.lowercaseSpan;
      // (I - I) in the same speaker's quotation.
      if(dict.firstPersonPronouns.contains(mString) && menRep.number==Number.SINGULAR
          && dict.firstPersonPronouns.contains(antString) && ant.number==Number.SINGULAR
          && Rules.entitySameSpeaker(document, menRep, ant)){
        document.errorLog.put(idPair, "same speaker for two <I>s");
        return true;
      }
      // (speaker - I)
      if(Rules.entityIsSpeaker(document, menRep, ant, dict) &&
          ((dict.firstPersonPronouns.contains(mString) && menRep.number==Number.SINGULAR)
              || (dict.firstPersonPronouns.contains(antString) && ant.number==Number.SINGULAR))) {
        document.errorLog.put(idPair, "speaker - I");
        return true;
      }
      if(Rules.entitySameSpeaker(document, menRep, ant)
          && dict.secondPersonPronouns.contains(mString)
          && dict.secondPersonPronouns.contains(antString)) {
        document.errorLog.put(idPair, "same speaker for two <you>s");
        return true;
      }
      // previous I - you or previous you - I in two person conversation
      if(((menRep.person==Person.I && ant.person==Person.YOU
          || (menRep.person==Person.YOU && ant.person==Person.I))
          && (menRep.headWord.get(UtteranceAnnotation.class)-ant.headWord.get(UtteranceAnnotation.class) == 1)
          && document.docType==DocType.CONVERSATION)) {
        SieveCoreferenceSystem.logger.finest("discourse match: between two person");
        document.errorLog.put(idPair, "you - I in conversation");
        return true;
      }
      if(dict.reflexivePronouns.contains(menRep.headString) && Rules.entitySubjectObject(menRep, ant)){
        SieveCoreferenceSystem.logger.finest("reflexive pronoun: "+ant.spanToString()+"("+ant.mentionID + ") :: "+ menRep.spanToString()+"("+menRep.mentionID + ") -> "+(menRep.goldCorefClusterID==ant.goldCorefClusterID));
        document.errorLog.put(idPair, "reflexive pronoun");
        return true;
      }
    }
    if(Constants.USE_DISCOURSE_CONSTRAINTS && !flags.USE_EXACTSTRINGMATCH && !flags.USE_RELAXED_EXACTSTRINGMATCH
        && !flags.USE_APPOSITION && !flags.USE_WORDS_INCLUSION && !flags.FOR_EVENT) {
      for(Mention m : menCluster.getCorefMentions()) {
        for(Mention a : antCluster.getCorefMentions()){
          if(Rules.entityIsSpeaker(document, m, a, dict) && m.person!=Person.I && a.person!=Person.I) {
            SieveCoreferenceSystem.logger.finest("Incompatibles: not match(speaker): " +ant.spanToString()+"("+ant.mentionID + ") :: "+ menRep.spanToString()+"("+menRep.mentionID + ") -> "+(menRep.goldCorefClusterID!=ant.goldCorefClusterID));
            document.incompatibles.add(new Pair<Integer, Integer>(Math.min(m.mentionID, a.mentionID), Math.max(m.mentionID, a.mentionID)));
            document.errorLog.put(idPair, "incompatible: not match speaker");
            return false;
          }
          int dist = Math.abs(m.headWord.get(UtteranceAnnotation.class) - a.headWord.get(UtteranceAnnotation.class));
          if(document.docType!=DocType.ARTICLE && dist==1 && !Rules.entitySameSpeaker(document, m, a)) {
            if(m.person==Person.I && a.person==Person.I) {
              SieveCoreferenceSystem.logger.finest("Incompatibles: neighbor I: " +ant.spanToString()+"("+ant.mentionID + ") :: "+ menRep.spanToString()+"("+menRep.mentionID + ") -> "+(menRep.goldCorefClusterID!=ant.goldCorefClusterID));
              document.incompatibles.add(new Pair<Integer, Integer>(Math.min(m.mentionID, a.mentionID), Math.max(m.mentionID, a.mentionID)));
              document.errorLog.put(idPair, "incompatible: neighbor <I>s");
              return false;
            }
            if(m.person==Person.YOU && a.person==Person.YOU) {
              SieveCoreferenceSystem.logger.finest("Incompatibles: neighbor YOU: " +ant.spanToString()+"("+ant.mentionID + ") :: "+ menRep.spanToString()+"("+menRep.mentionID + ") -> "+(menRep.goldCorefClusterID!=ant.goldCorefClusterID));
              document.incompatibles.add(new Pair<Integer, Integer>(Math.min(m.mentionID, a.mentionID), Math.max(m.mentionID, a.mentionID)));
              document.errorLog.put(idPair, "incompatible: neighbor <you>");
              return false;
            }
            if(m.person==Person.WE && a.person==Person.WE) {
              SieveCoreferenceSystem.logger.finest("Incompatibles: neighbor WE: " +ant.spanToString()+"("+ant.mentionID + ") :: "+ menRep.spanToString()+"("+menRep.mentionID + ") -> "+(menRep.goldCorefClusterID!=ant.goldCorefClusterID));
              document.incompatibles.add(new Pair<Integer, Integer>(Math.min(m.mentionID, a.mentionID), Math.max(m.mentionID, a.mentionID)));
              document.errorLog.put(idPair, "incompatible: neighbor <we>");
              return false;
            }
          }
        }
      }
      if(document.docType==DocType.ARTICLE) {
        for(Mention m : menCluster.getCorefMentions()) {
          for(Mention a : antCluster.getCorefMentions()){
            if(Rules.entitySubjectObject(m, a)) {
              SieveCoreferenceSystem.logger.finest("Incompatibles: subject-object: "+ant.spanToString()+"("+ant.mentionID + ") :: "+ menRep.spanToString()+"("+menRep.mentionID + ") -> "+(menRep.goldCorefClusterID!=ant.goldCorefClusterID));
              document.incompatibles.add(new Pair<Integer, Integer>(Math.min(m.mentionID, a.mentionID), Math.max(m.mentionID, a.mentionID)));
              document.errorLog.put(idPair, "incompatible: subj - obj");
              return false;
            }
          }
        }
      }
    }
    if(flags.ENTITY_PREDICATE_MATCH && Rules.entityHeadsAgree(menCluster, antCluster, menRep, ant, dict) 
        && Rules.entityHaveSameStringPredicate(menCluster, antCluster, document)) {
      return true;
    }
    if(flags.ENTITY_MULTIPLE_ARG_MATCH && Rules.entityMultipleArgCoref(menCluster, antCluster, document, flags.ENTITY_ARG_MATCH_THRES)) {
      document.errorLog.put(idPair, "entityMultipleArgCoref: multiple arguments matched");
      return true;
    }
    if(flags.USE_THESAURUS_ENTITY_SIMILAR && Rules.entityThesaurusSimilar(menCluster, antCluster, flags.THESAURUS_ENTITY_SIMILAR_THRESHOLD, document)) {
      document.errorLog.put(idPair, "entityDekangLinSimilar: similar in thesaurus");
      ret = true;
    }
    if(flags.USE_COREF_VERB_FOR_ENTITY && Rules.entityThesaurusSimilarWithCorefVerb(menCluster, antCluster, flags.THESAURUS_ENTITY_SIMILAR_THRESHOLD, document)) {
      document.errorLog.put(idPair, "entityThesaurusSimilarWithCorefVerb: similar in thesaurus & have coref verbs");
      ret = true;
    }

    if(flags.COREF_PRED && mention.predicate!=null && ant.predicate!=null
        && mention.predicate.corefClusterID!=ant.predicate.corefClusterID
        && mention.isSubject==ant.isSubject && mention.isDirectObject==ant.isDirectObject
        && mention.isIndirectObject==ant.isIndirectObject) {
      document.errorLog.put(idPair, "same role for not-coreferent predicates");
      return false;
    }
    if(flags.USE_COREF_SRLPREDICATE_MATCH && Rules.entityHaveCorefSrlPredicate(menCluster, antCluster)
        //    if(flags.USE_COREF_SRLPREDICATE_MATCH && Rules.entitySameSrlPredicate(mention, ant)
        && Rules.entityThesaurusSimilar(menCluster, antCluster, flags.THESAURUS_ENTITY_SIMILAR_THRESHOLD, document)) {
      document.errorLog.put(idPair, "a pair of mentions from each cluster satisfied: in the same role with coref predicates, attributes agree, at least 1 same content word");
      ret = true;
    }
    if(flags.COREF_SRLPRED && !Rules.entityHaveCorefSrlPredicate(menCluster, antCluster)) {
      document.errorLog.put(idPair, "no mention pair from clusters satisfied: in the same role with coref predicates");
      return false;
    }
    if(flags.SIMILAR_SRLPRED && !Rules.entityHaveSimilarSrlPredicate(menCluster, antCluster, document)) {
      document.errorLog.put(idPair, "no mention pair from clusters satisfied: in the same role with similar predicates");
      return false;
    }

    if(flags.USE_iwithini && Rules.entityIWithinI(menRep, ant, dict)) {
      document.incompatibles.add(new Pair<Integer, Integer>(Math.min(menRep.mentionID, ant.mentionID), Math.max(menRep.mentionID, ant.mentionID)));
      document.errorLog.put(idPair, "i-within-i");
      return false;
    }
    if(flags.USE_EXACTSTRINGMATCH && Rules.entityExactStringMatch(menCluster, antCluster, dict, roleSet)){
      //    if(flags.USE_EXACTSTRINGMATCH && Rules.entityExactStringMatch(mention, ant, roleSet)){
      document.errorLog.put(idPair, "exact string match");
      return true;
    }
    if(flags.USE_RELAXED_EXACTSTRINGMATCH && Rules.entityRelaxedExactStringMatch(menCluster, antCluster, menRep, ant, dict, roleSet)){
      document.errorLog.put(idPair, "relaxed exact string match");
      return true;
    }
    if(flags.USE_APPOSITION && Rules.entityIsApposition(menCluster, antCluster, menRep, ant)) {
      SieveCoreferenceSystem.logger.finest("Apposition: "+menRep.spanToString()+"\tvs\t"+ant.spanToString());
      document.errorLog.put(idPair, "apposition");
      return true;
    }
    if(flags.USE_PREDICATENOMINATIVES && Rules.entityIsPredicateNominatives(menCluster, antCluster, menRep, ant)) {
      SieveCoreferenceSystem.logger.finest("Predicate nominatives: "+menRep.spanToString()+"\tvs\t"+ant.spanToString());
      document.errorLog.put(idPair, "predicate nominatives");
      return true;
    }

    if(flags.USE_ACRONYM && Rules.entityIsAcronym(menCluster, antCluster)) {
      SieveCoreferenceSystem.logger.finest("Acronym: "+menRep.spanToString()+"\tvs\t"+ant.spanToString());
      document.errorLog.put(idPair, "acronym");
      return true;
    }
    if(flags.USE_RELATIVEPRONOUN && Rules.entityIsRelativePronoun(menRep, ant)){
      SieveCoreferenceSystem.logger.finest("Relative pronoun: "+menRep.spanToString()+"\tvs\t"+ant.spanToString());
      document.errorLog.put(idPair, "relative pronoun");
      return true;
    }
    if(flags.USE_DEMONYM && menRep.isDemonym(ant, dict)){
      SieveCoreferenceSystem.logger.finest("Demonym: "+menRep.spanToString()+"\tvs\t"+ant.spanToString());
      document.errorLog.put(idPair, "demonym");
      return true;
    }

    if(flags.USE_ROLEAPPOSITION && Rules.entityIsRoleAppositive(menCluster, antCluster, menRep, ant, dict)){
      document.errorLog.put(idPair, "role appositive");
      return true;
    }
    if(flags.USE_INCLUSION_HEADMATCH && Rules.entityHeadsAgree(menCluster, antCluster, menRep, ant, dict)){
      document.errorLog.put(idPair, "heads agree");
      ret = true;
    }
    if(flags.USE_RELAXED_HEADMATCH && Rules.entityRelaxedHeadsAgreeBetweenMentions(menCluster, antCluster, menRep, ant) ){
      document.errorLog.put(idPair, "relaxed heads agree");
      ret = true;
    }
    if(flags.USE_WORDS_INCLUSION && ret && ! Rules.entityWordsIncluded(menCluster, antCluster, menRep, ant)) {
      document.errorLog.put(idPair, "not words inclusion");
      return false;
    }

    if(flags.USE_INCOMPATIBLE_MODIFIER && ret && Rules.entityHaveIncompatibleModifier(menCluster, antCluster)) {
      document.errorLog.put(idPair, "have incompatible modifier");
      return false;
    }
    if(flags.USE_PROPERHEAD_AT_LAST && ret && !Rules.entitySameProperHeadLastWord(menCluster, antCluster, menRep, ant)) {
      document.errorLog.put(idPair, "not same proper headword at last");
      return false;
    }
    if(flags.USE_ATTRIBUTES_AGREE && !Rules.entityAttributesAgree(menCluster, antCluster)) {
      document.errorLog.put(idPair, "not attributes agree");
      return false;
    }
    if(flags.USE_DIFFERENT_LOCATION
        && Rules.entityHaveDifferentLocation(menRep, ant, dict)) {
      if(flags.USE_PROPERHEAD_AT_LAST  && ret && menRep.goldCorefClusterID!=ant.goldCorefClusterID) {
        SieveCoreferenceSystem.logger.finest("DIFFERENT LOCATION: "+ant.spanToString()+" :: "+menRep.spanToString());
      }
      document.errorLog.put(idPair, "have different locations");
      return false;
    }
    if(flags.USE_NUMBER_IN_MENTION
        && Rules.entityNumberInLaterMention(menRep, ant)) {
      if(flags.USE_PROPERHEAD_AT_LAST  && ret && menRep.goldCorefClusterID!=ant.goldCorefClusterID) {
        SieveCoreferenceSystem.logger.finest("NEW NUMBER : "+ant.spanToString()+" :: "+menRep.spanToString());
      }
      document.errorLog.put(idPair, "have new number in later mention");
      return false;
    }
    if(flags.USE_WN_HYPERNYM) {
      if(semantics.wordnet.checkHypernym(menRep, ant)) {
        document.errorLog.put(idPair, "WN hypernym");
        ret = true;
      } else if (menRep.goldCorefClusterID == ant.goldCorefClusterID
          && !menRep.isPronominal() && !ant.isPronominal()){
        SieveCoreferenceSystem.logger.finest("not hypernym in WN");
        SieveCoreferenceSystem.logger.finest("False Negatives:: " + ant.spanToString() +" <= "+menRep.spanToString());
      }
    }
    if(flags.USE_WN_SYNONYM) {
      if(semantics.wordnet.checkSynonym(menRep, ant)) {
        document.errorLog.put(idPair, "WN synonym");
        ret = true;
      } else if (menRep.goldCorefClusterID == ant.goldCorefClusterID
          && !menRep.isPronominal() && !ant.isPronominal()){
        SieveCoreferenceSystem.logger.finest("not synonym in WN");
        SieveCoreferenceSystem.logger.finest("False Negatives:: " + ant.spanToString() +" <= "+menRep.spanToString());

      }
    }

    try {
      if(flags.USE_ALIAS && Rules.entityAlias(menCluster, antCluster, semantics, dict)){
        document.errorLog.put(idPair, "alias");
        return true;
      }
    } catch (Exception e) {
      throw new RuntimeException(e);
    }
    
    if(flags.USE_DISTANCE && Rules.entityTokenDistance(mention, ant)){
      return false;
    }
    
    if(flags.USE_NUMBER_ANIMACY_NE_AGREE && !Rules.entityNumberAnimacyNEAgree(menCluster, antCluster)){
      return false;
    }
    
    if(flags.USE_COREF_DICT){

      // Head match
      if(ant.headWord.lemma().equals(mention.headWord.lemma())) return false;
      
      // Constraint: ignore pairs commonNoun - properNoun
      if(!ant.isNE() && 
         ( mention.headWord.get(PartOfSpeechAnnotation.class).startsWith("NNP") 
           || !mention.headWord.word().substring(1).equals(mention.headWord.word().substring(1).toLowerCase()) ) ) return false;      
      
      // Constraint: ignore plurals
      if(ant.headWord.get(PartOfSpeechAnnotation.class).equals("NNS")
          && mention.headWord.get(PartOfSpeechAnnotation.class).equals("NNS")) return false;

      // Constraint: ignore pre- and post-modified plurals
      /*if( (ant.originalSpan.size() != 1 || mention.originalSpan.size() != 1 )
          && ant.headWord.get(PartOfSpeechAnnotation.class).equals("NNS")
          && mention.headWord.get(PartOfSpeechAnnotation.class).equals("NNS")) return false;*/

      // Constraint: ignore post-modified plurals
      /*if( (ant.originalSpan.size() > ant.originalSpan.indexOf(ant.headWord)+1 
           || mention.originalSpan.size() > mention.originalSpan.indexOf(mention.headWord)+1) 
          && ant.headWord.get(PartOfSpeechAnnotation.class).equals("NNS")
          && mention.headWord.get(PartOfSpeechAnnotation.class).equals("NNS")) return false;*/
     
      // Constraint: ignore mentions with indefinite determiners
      if(ant.hasIndefDT() || mention.hasIndefDT()) return false;  
      
      // Constraint: ignore coordinated mentions
      if(ant.isCoordinated() || mention.isCoordinated()) return false;

      // Constraint: context incompatibility
      if(Rules.contextIncompatible(mention, ant, dict)) return false;

      // Constraint: sentence context incompatibility when the mentions are common nouns
      if(Rules.sentenceContextIncompatible(mention, ant, dict)) return false;
      
      if(flags.USE_COREF_DICT_COL1       
          && Rules.entityClusterAllCorefDictionary(menCluster, antCluster, dict, 1, 8)) { 
        
        if (menRep.goldCorefClusterID == ant.goldCorefClusterID){
          System.out.println("True positive. Col1: "+ant+" // "+mention);
          System.out.println("Sent-ant: "+ Mention.sentenceWordsToString(ant));
          System.out.println("Sent-men: "+ Mention.sentenceWordsToString(mention)+"\n");

        } else {
          System.out.println("False positive. Col1: "+ant+" // "+mention);
          System.out.println("Sent-ant: "+ Mention.sentenceWordsToString(ant));
          System.out.println("Sent-men: "+ Mention.sentenceWordsToString(mention)+"\n");
        }
        return true;       
      }
      
      if(flags.USE_COREF_DICT_COL2 
          && Rules.entityCorefDictionary(menRep, ant, dict, 2, 2)) { 
        
        if (menRep.goldCorefClusterID == ant.goldCorefClusterID){
          System.out.println("True positive. Col2: "+ant+" // "+mention);
          System.out.println("Sent-ant: "+ Mention.sentenceWordsToString(ant));
          System.out.println("Sent-men: "+ Mention.sentenceWordsToString(mention)+"\n");

        } else {
          System.out.println("False positive. Col2: "+ant+" // "+mention);
          System.out.println("Sent-ant: "+ Mention.sentenceWordsToString(ant));
          System.out.println("Sent-men: "+ Mention.sentenceWordsToString(mention)+"\n");
        }
        return true;       
      }
      
      if(flags.USE_COREF_DICT_COL3 
          && Rules.entityCorefDictionary(menRep, ant, dict, 3, 2)) {
        
        if (menRep.goldCorefClusterID == ant.goldCorefClusterID){
          System.out.println("True positive. Col3: "+ant+" // "+mention);
          System.out.println("Sent-ant: "+ Mention.sentenceWordsToString(ant));
          System.out.println("Sent-men: "+ Mention.sentenceWordsToString(mention)+"\n");
        } else {
          System.out.println("False positive. Col3: "+ant+" // "+mention);
          System.out.println("Sent-ant: "+ Mention.sentenceWordsToString(ant));
          System.out.println("Sent-men: "+ Mention.sentenceWordsToString(mention)+"\n");
        }
        return true;       
      }
      
      if(flags.USE_COREF_DICT_COL4 
          && Rules.entityCorefDictionary(menRep, ant, dict, 4, 2)) { 
        
        if (menRep.goldCorefClusterID == ant.goldCorefClusterID){
          System.out.println("True positive. Col4: "+ant+" // "+mention);
          System.out.println("Sent-ant: "+ Mention.sentenceWordsToString(ant));
          System.out.println("Sent-men: "+ Mention.sentenceWordsToString(mention)+"\n");
        } else {
          System.out.println("False positive. Col4: "+ant+" // "+mention);
          System.out.println("Sent-ant: "+ Mention.sentenceWordsToString(ant));
          System.out.println("Sent-men: "+ Mention.sentenceWordsToString(mention)+"\n");
        }
        return true;       
      }    
    }
    
    if(flags.DO_PRONOUN){
      Mention m;
      if (menRep.predicateNominatives!=null && menRep.predicateNominatives.contains(mention)) {
        m = mention;
      } else {
        m = menRep;
      }

      if((m.isPronominal() || dict.allPronouns.contains(m.toString())) && Rules.entityAttributesAgree(menCluster, antCluster)){

        if(dict.demonymSet.contains(ant.lowercaseSpan) && dict.notOrganizationPRP.contains(m.headString)){
          document.incompatibles.add(new Pair<Integer, Integer>(Math.min(m.mentionID, ant.mentionID), Math.max(m.mentionID, ant.mentionID)));
          document.errorLog.put(idPair, "incompatible pronoun");
          return false;
        }
        if(Constants.USE_DISCOURSE_CONSTRAINTS && Rules.entityPersonDisagree(document, menCluster, antCluster, dict)){
          SieveCoreferenceSystem.logger.finest("Incompatibles: Person Disagree: "+ant.spanToString()+"("+ant.mentionID+") :: "+menRep.spanToString()+"("+menRep.mentionID+") -> "+(menRep.goldCorefClusterID!=ant.goldCorefClusterID));
          document.incompatibles.add(new Pair<Integer, Integer>(Math.min(m.mentionID, ant.mentionID), Math.max(m.mentionID, ant.mentionID)));
          document.errorLog.put(idPair, "person disagree");
          return false;
        }
        document.errorLog.put(idPair, "pronoun match");
        return true;
      }
    }

    return ret;
  }
  private boolean coreferentEvent(Document document, CorefCluster menCluster, CorefCluster antCluster, Mention mention, Mention ant, Dictionaries dict, Semantics semantics) {

    //    if(flags.EVENT_LOCATION && !Rules.eventSameLocation(mention, ant)) {
    //      return false;
    //    }
    //    if(flags.EVENT_TIME && !Rules.eventSameTime(mention, ant)) {
    //      return false;
    //    }
    //    Map<Integer, Mention> golds = document.allGoldMentions;
    //    if(golds.containsKey(mention.mentionID) && golds.containsKey(ant.mentionID)) {
    //      if((flags.ORACLE_EVENT && golds.get(mention.mentionID).isEvent && golds.get(ant.mentionID).isEvent)
    //          ) {
    //        //      if(!mention2.isEvent && !ant.isEvent) {
    //        if(golds.containsKey(mention.mentionID) && golds.containsKey(ant.mentionID)) {
    //          if(golds.get(mention.mentionID).goldCorefClusterID == golds.get(ant.mentionID).goldCorefClusterID) {
    //            return true;
    //          }
    //          return false;
    //        }
    //      }
    //    }
    if(!mention.isVerb && !ant.isVerb) return false;

    boolean ret = false;
    IntPair idPair = new IntPair(Math.min(mention.mentionID, ant.mentionID), Math.max(mention.mentionID, ant.mentionID));

    //
    // general rules: applied to all event sieves
    //

    // reporting mention rule: report and non-report mentions cannot be coreferent
    //    if(mention.isReport!=ant.isReport) {
    //      document.errorLog.put(idPair, "only one is report mention");
    //      return false;
    //    }

    //
    // sieve specific rules
    //

    if(flags.EVENT_ARGUMENT_MATCH && Rules.eventSameLemma(menCluster, antCluster) 
        && Rules.eventHaveSameStringArgs(menCluster, antCluster, document)) {
      return true;
    }
    if(flags.EVENT_MULTIPLE_ARG_MATCH && Rules.eventMultipleArgCoref(menCluster, antCluster, document, flags.EVENT_ARG_MATCH_THRES)) {
      ret = true;
    }
    if(flags.EVENT_LOCATION && !Rules.eventSameLocation(mention, ant)) {
      return false;
    }
    if(flags.EVENT_TIME && !Rules.eventSameTime(mention, ant)) {
      return false;
    }
    if(flags.USE_NOUNEVENTS_MATCH_ONLY && mention.isVerb && ant.isVerb) {
      return false;
    }
    if(flags.MATCH_NOSRLARGEVENT_ONLY && (menCluster.srlRoles.size()!=0 && antCluster.srlRoles.size()!=0)) {
      return false;
    }
    if(flags.USE_NO_COMMON_SRLARG_MATCH && Rules.eventHaveCommonSrlArg(menCluster, antCluster)) {
      return false;
    }
    if(flags.CLUSTER_SYNONYM_THRESHOLD!=-1 && !Rules.eventClusterSynonym(menCluster, antCluster, flags.CLUSTER_SYNONYM_THRESHOLD, document)) {
      document.errorLog.put(idPair, "eventClusterSynonym: not synonym");
      return false;
    }
    if(flags.CLUSTER_SIMILAR_THRESHOLD!=-1 && !Rules.eventClusterSimilar(menCluster, antCluster, flags.CLUSTER_SIMILAR_THRESHOLD, document)) {
      document.errorLog.put(idPair, "eventClusterSimilar: not similar");
      return false;
    }
    if(flags.USE_SRLARGUMENT_MATCH && Rules.eventSrlArgumentMatch(menCluster, antCluster, document)) {
      document.errorLog.put(idPair, "eventSrlArgumentMatch");
      ret = true;
    }
    if(flags.USE_DEKANGLIN_SIMPLE && Rules.eventDekangLinSimpleMatch(mention, ant, dict)) {
      document.errorLog.put(idPair, "DekangLin SimpleMatch");
      ret = true;
    }

    if(flags.USE_SUPERSENSE && Rules.eventSuperSenseMatch(mention, ant, dict)) {
      document.errorLog.put(idPair, "SuperSense Match");
      ret = true;
    }

    if(flags.USE_SCHEMA_MATCH && Rules.eventSameSchema(mention, ant, dict)) {
      RuleBasedJointCorefSystem.logger.finer("same schema: "+mention.toString()+" "+mention.mentionID+ " <-> "+ant.toString()+" "+ant.mentionID);
      document.errorLog.put(idPair, "Same Schema");
      ret = true;
    }
    if(flags.USE_WNSYNSET_CLUSTER
        && WordNet.sameSynsetCluster(mention, ant, semantics.wordnet)) {
      //        && WordNet.sameSynsetCluster(mention, ant, semantics.wordnet)) {
      RuleBasedJointCorefSystem.logger.finer("same WN synset cluster: "+mention.toString()+" "+mention.mentionID+ " <-> "+ant.toString()+" "+ant.mentionID);
      document.errorLog.put(idPair, "Same WN synset cluster (transitive closure)");
      ret = true;
    }
    if(flags.USE_WNSYNONYM
        && WordNet.synonymInWN(mention, ant, semantics.wordnet, dict)) {
      RuleBasedJointCorefSystem.logger.finer("WN synonym: "+mention.toString()+" "+mention.mentionID+ " <-> "+ant.toString()+" "+ant.mentionID);
      document.errorLog.put(idPair, "WN synonym");
      ret = true;
    }
    if(flags.USE_WNSIMILARITY_CLUSTER
        && Rules.eventSimilarInWN(menCluster, antCluster, flags.WNSIMILARITY_CLUSTER_THRESHOLD, document)) {
      //        && WordNet.similarInWN(menCluster, antCluster, flags.WNSIMILARITY_CLUSTER_THRESHOLD, semantics.wordnet, dict)) {

      RuleBasedJointCorefSystem.logger.finer("WN similar cluster: "+mention.toString()+" "+mention.mentionID+ " <-> "+ant.toString()+" "+ant.mentionID);
      document.errorLog.put(idPair, "WN similar cluster");
      ret = true;
    }
    if(flags.USE_WNSIMILARITY_MENTION
        && WordNet.similarInWN(mention, ant, semantics.wordnet, dict)) {
      RuleBasedJointCorefSystem.logger.finer("WN similar mention: "+mention.toString()+" "+mention.mentionID+ " <-> "+ant.toString()+" "+ant.mentionID);
      document.errorLog.put(idPair, "WN similar mention");
      ret = true;
    }
    if(flags.USE_WNSIMILARITY_SURFACE_CONTEXT && Rules.eventWNSimilaritySurfaceContext(menCluster, antCluster, semantics.wordnet, document, dict)) {
      ret = true;
    }
    if(flags.USE_LEMMAMATCH && Rules.eventSameLemma(menCluster, antCluster)) {
      RuleBasedJointCorefSystem.logger.finer("lemma matched: "+mention.toString() + " <-> "+ant.toString());
      document.errorLog.put(idPair, "Lemma match");
      ret = true;
    }
//    if(flags.USE_LEMMAMATCH && Rules.eventSameLemma(mention, ant)) {
//      RuleBasedJointCorefSystem.logger.finer("lemma matched: "+mention.toString() + " <-> "+ant.toString());
//      document.errorLog.put(idPair, "Lemma match");
//      ret = true;
//    }
    if(flags.USE_LEMMA_OBJ && Rules.eventLemmaObj(menCluster, antCluster, mention, ant, document, dict)) {
      RuleBasedJointCorefSystem.logger.finer("lemmaObj: "+mention.toString() + " <-> "+ant.toString());
      document.errorLog.put(idPair, "Lemma-Obj Match (the lemma of a verb event == another's object)");
      ret = true;
    }
    if(flags.USE_SENT_SIMILAR && Rules.eventSentSimilar(menCluster, antCluster, document)) {
      document.errorLog.put(idPair, "eventSentSimilar: sentences similar");
      ret = true;
    }

    // i-within-i rule: an event mention is dominated the other's -> cannot be coreferent
    if(flags.USE_EVENT_iwithini && Rules.eventIWithinI(menCluster, antCluster, document)) {
      document.errorLog.put(idPair, "eventIWithinI: i-within-i");
      return false;
    }
    if(flags.USE_NOT_FREQ_VERB && (Rules.eventFrequentVerb(mention, dict, flags.FREQ_IDF_CUTOFF) || Rules.eventFrequentVerb(ant, dict, flags.FREQ_IDF_CUTOFF))) {
      document.errorLog.put(idPair, "eventFrequentVerb: frequent verb");
      return false;
    }
    if(flags.USE_ARG_SHARE && ret && !Rules.eventShareArgument(mention, ant)) {
      RuleBasedJointCorefSystem.logger.finer("no shared argument: "+mention.toString() + " <-> "+ant.toString());
      document.errorLog.put(idPair, "eventShareArgument: no shared argument");
      ret = false;
    }
    if(flags.USE_COREF_ARG && ret && !Rules.eventHaveCorefArgument(mention, ant, document)) {
      RuleBasedJointCorefSystem.logger.finer("no coref argument: "+mention.toString() + " <-> "+ant.toString());
      document.errorLog.put(idPair, "eventHaveCorefArgument: no coref argument");
      ret = false;
    }
    if(flags.COREF_SRLARG && ret && !Rules.eventHaveCorefSrlArgument(menCluster, antCluster)) {
      RuleBasedJointCorefSystem.logger.finer("srl coref argument: "+mention.toString() + " <-> "+ant.toString());
      document.errorLog.put(idPair, "eventHaveCorefSrlArgument: no srl coref argument");
      ret = false;
    }
    if(flags.SIMILAR_SRLARG && true) {
      // TODO later (check same head first)
      // this uses thesaurus similar
    }
    if(flags.SAMEHEAD_SRLARG && ret && !Rules.eventHaveSameHeadSrlArgument(menCluster, antCluster)) {
      document.errorLog.put(idPair, "eventHaveSameHeadSrlArgument: no srl samehead argument");
      ret = false;
    }
    if(flags.USE_NOT_COREF_SRLARG && ret && Rules.eventHaveNotCorefSrlArgument(menCluster, antCluster)) {
      RuleBasedJointCorefSystem.logger.finer("not coref srl argument: "+mention.toString() + " <-> "+ant.toString());
      document.errorLog.put(idPair, "eventHaveNotCorefSrlArgument: not coref srl argument");
      ret = false;
    }
    if(flags.USE_POSSIBLE_COREF_SRLARG && ret && !Rules.eventHavePossibleCorefSrlArgument(menCluster, antCluster, dict)) {
      //      if(flags.USE_RELAXED_COREF_SRLARG && ret && !Rules.eventHavePossibleCorefSrlArgument(mention, ant, dict)) {
      document.errorLog.put(idPair, "eventHavePossibleCorefSrlArgument: no possible srl coref argument");
      ret = false;
    }
    if(flags.USE_COREF_IN_SENTENCE && ret && !Rules.eventHaveCorefEntityInSentence(mention, ant, document)) {
      RuleBasedJointCorefSystem.logger.finer("no coref entities in sentence: "+mention.toString() + " <-> "+ant.toString());
      document.errorLog.put(idPair, "eventHaveCorefEntityInSentence: no coref entities in sentences");
      ret = false;
    }
    if(flags.USE_NOT_COREF_ARG && ret && Rules.eventHaveNotCorefArg(menCluster, antCluster, mention, ant, document, semantics.wordnet, dict)){
      RuleBasedJointCorefSystem.logger.finer("not coref entities in argument: "+mention.toString() + " <-> "+ant.toString());
      document.errorLog.put(idPair, "eventHaveNotCorefArg: not coref entities in arguments");
      ret = false;
    }
    if(flags.USE_THESAURUS_SIMILAR_SRLARG && ret && !Rules.eventThesaurusSimilarSrlArg(menCluster, antCluster, flags.THESAURUS_SIMILAR_THRESHOLD, document)){
      document.errorLog.put(idPair, "eventThesaurusSimilarSrlArg: not similar args");
      ret = false;
    }
    if(flags.USE_COREF_LEFTMENTION && ret && !Rules.eventHaveCorefLeftMention(mention, ant, document)) {
      ret = false;
    }
    if(flags.USE_COREF_RIGHTMENTION && ret && !Rules.eventHaveCorefRightMention(mention, ant, document)) {
      ret = false;
    }

    return ret;
  }
  /**
   * Orders the antecedents for the given mention (m1)
   * @param antecedentSentence
   * @param mySentence
   * @param orderedMentions
   * @param orderedMentionsBySentence
   * @param m1
   * @param m1Position
   * @param corefClusters
   * @param dict
   * @return
   */
  public List<Mention> getOrderedAntecedents(
      int antecedentSentence,
      int mySentence,
      List<Mention> orderedMentions,
      List<List<Mention>> orderedMentionsBySentence,
      Mention m1,
      int m1Position,
      Map<Integer, CorefCluster> corefClusters,
      Dictionaries dict) {
    List<Mention> orderedAntecedents = new ArrayList<Mention>();

    // ordering antecedents
    if (antecedentSentence == mySentence) {   // same sentence
      orderedAntecedents.addAll(orderedMentions.subList(0, m1Position));
      if(flags.DO_PRONOUN && corefClusters.get(m1.corefClusterID).isSinglePronounCluster(dict)) {
        orderedAntecedents = sortMentionsForPronoun(orderedAntecedents, m1, true);
      }
      if(dict.relativePronouns.contains(m1.spanToString())) Collections.reverse(orderedAntecedents);
    } else {    // previous sentence
      orderedAntecedents.addAll(orderedMentionsBySentence.get(antecedentSentence));
    }

    return orderedAntecedents;
  }

  /** Divides a sentence into clauses and sort the antecedents for pronoun matching  */
  private List<Mention> sortMentionsForPronoun(List<Mention> l, Mention m1, boolean sameSentence) {
    List<Mention> sorted = new ArrayList<Mention>();
    Tree tree = m1.contextParseTree;
    Tree current = m1.mentionSubTree;
    if(sameSentence){
      while(true){
        current = current.ancestor(1, tree);
        if(current.label().value().startsWith("S")){
          for(Mention m : l){
            if(!sorted.contains(m) && current.dominates(m.mentionSubTree)) sorted.add(m);
          }
        }
        if(current.label().value().equals("ROOT") || current.ancestor(1, tree)==null) break;
      }
      if(l.size()!=sorted.size()) {
        SieveCoreferenceSystem.logger.finest("sorting failed!!! -> parser error?? \tmentionID: "+m1.mentionID+" " + m1.spanToString());
        sorted=l;
      } else if(!l.equals(sorted)){
        SieveCoreferenceSystem.logger.finest("sorting succeeded & changed !! \tmentionID: "+m1.mentionID+" " + m1.spanToString());
        for(int i=0; i<l.size(); i++){
          Mention ml = l.get(i);
          Mention msorted = sorted.get(i);
          SieveCoreferenceSystem.logger.finest("\t["+ml.spanToString()+"]\t["+msorted.spanToString()+"]");
        }
      } else {
        SieveCoreferenceSystem.logger.finest("no changed !! \tmentionID: "+m1.mentionID+" " + m1.spanToString());
      }
    }
    return sorted;
  }
}



