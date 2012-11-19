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
import java.util.EnumSet;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.logging.Logger;

import edu.stanford.nlp.jcoref.dcoref.Dictionaries.Animacy;
import edu.stanford.nlp.jcoref.dcoref.Dictionaries.Gender;
import edu.stanford.nlp.jcoref.dcoref.Dictionaries.Number;
import edu.stanford.nlp.ling.CoreAnnotations.TextAnnotation;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.stats.ClassicCounter;
import edu.stanford.nlp.stats.Counter;

/**
 * One cluster for the SieveCoreferenceSystem.
 *
 * @author Heeyoung Lee
 */
public class CorefCluster implements Serializable{

  private static final long serialVersionUID = 8655265337578515592L;
  protected Set<Mention> corefMentions;
  protected int clusterID;

  // Attributes for cluster - can include multiple attribute e.g., {singular, plural}
  protected Set<Number> numbers;
  protected Set<Gender> genders;
  protected Set<Animacy> animacies;
  protected Set<String> nerStrings;
  protected Set<String> heads;

  /** All words in this cluster - for word inclusion feature  */
  public Set<String> words;

  /** The first mention in this cluster */
  protected Mention firstMention;

  /** Return the most representative mention in the chain.
   *  Proper mention and a mention with more pre-modifiers are preferred.
   */
  protected Mention representative;

  public int getClusterID(){ return clusterID; }
  public Set<Mention> getCorefMentions() { return corefMentions; }
  public Mention getFirstMention() { return firstMention; }
  public Mention getRepresentativeMention() { return representative; }

  public Map<String, Set<Mention>> srlRoles;
  public Map<String, Set<Mention>> srlPredicates;
  
  public HashMap<String, ClassicCounter<String>> predictedCentroid = new HashMap<String, ClassicCounter<String>>();
  public HashMap<String, ClassicCounter<String>> goldCentroid = new HashMap<String, ClassicCounter<String>>();

  public CorefCluster(int ID) {
    clusterID = ID;
    corefMentions = new HashSet<Mention>();
    numbers = EnumSet.noneOf(Number.class);
    genders = EnumSet.noneOf(Gender.class);
    animacies = EnumSet.noneOf(Animacy.class);
    nerStrings = new HashSet<String>();
    heads = new HashSet<String>();
    words = new HashSet<String>();
    firstMention = null;
    representative = null;
    srlRoles = new HashMap<String, Set<Mention>>();
    srlPredicates = new HashMap<String, Set<Mention>>();
  }

  public CorefCluster(){
    this(-1);
  }

  public CorefCluster(int ID, Set<Mention> mentions){
    this(ID);
    corefMentions.addAll(mentions);
    for(Mention m : mentions){
      animacies.add(m.animacy);
      genders.add(m.gender);
      numbers.add(m.number);
      nerStrings.add(m.nerString);
      if(!m.isPronominal()){
        heads.add(m.headString);
        for(CoreLabel w : m.originalSpan){
          words.add(w.get(TextAnnotation.class).toLowerCase());
        }
      }
      if(firstMention == null) firstMention = m;
      else {
        if(m.appearEarlierThan(firstMention)) firstMention = m;
      }
      for(String role : m.srlArgs.keySet()) {
        if(!srlRoles.containsKey(role)) srlRoles.put(role, new HashSet<Mention>());
        srlRoles.get(role).add(m.srlArgs.get(role));
      }
      for(Mention predicate : m.srlPredicates.keySet()) {
        String role = m.srlPredicates.get(predicate);
        if(!srlPredicates.containsKey(role)) srlPredicates.put(role, new HashSet<Mention>());
        srlPredicates.get(role).add(predicate);
      }

    }
    representative = firstMention;
    for(Mention m : mentions) {
      if(m.moreRepresentativeThan(representative)) representative = m;
    }
  }
  public String toString() {
    return corefMentions.toString();
  }

  /** merge 2 clusters: to = to + from */
  public static void mergeClusters(CorefCluster to, CorefCluster from) {
    int toID = to.clusterID;
    for (Mention m : from.corefMentions){
      m.corefClusterID = toID;
    }
    if(Constants.SHARE_ATTRIBUTES){
      to.numbers.addAll(from.numbers);
      if(to.numbers.size() > 1 && to.numbers.contains(Number.UNKNOWN)) {
        to.numbers.remove(Number.UNKNOWN);
      }

      to.genders.addAll(from.genders);
      if(to.genders.size() > 1 && to.genders.contains(Gender.UNKNOWN)) {
        to.genders.remove(Gender.UNKNOWN);
      }

      to.animacies.addAll(from.animacies);
      if(to.animacies.size() > 1 && to.animacies.contains(Animacy.UNKNOWN)) {
        to.animacies.remove(Animacy.UNKNOWN);
      }

      to.nerStrings.addAll(from.nerStrings);
      if(to.nerStrings.size() > 1 && to.nerStrings.contains("O")) {
        to.nerStrings.remove("O");
      }
      if(to.nerStrings.size() > 1 && to.nerStrings.contains("MISC")) {
        to.nerStrings.remove("MISC");
      }
    }

    to.heads.addAll(from.heads);
    to.corefMentions.addAll(from.corefMentions);
    to.words.addAll(from.words);
    if(from.firstMention.appearEarlierThan(to.firstMention) && !from.firstMention.isPronominal()) to.firstMention = from.firstMention;
    if(from.representative.moreRepresentativeThan(to.representative)) to.representative = from.representative;
    SieveCoreferenceSystem.logger.finer("merge clusters: "+toID+" += "+from.clusterID);
    for(String role : from.srlRoles.keySet()) {
      Set<Mention> fromSrlRoles = from.srlRoles.get(role);
      if(to.srlRoles.containsKey(role)) to.srlRoles.get(role).addAll(fromSrlRoles);
      else to.srlRoles.put(role, fromSrlRoles);
    }
    for(String role : from.srlPredicates.keySet()) {
      Set<Mention> fromSrlPredicates = from.srlPredicates.get(role);
      if(to.srlPredicates.containsKey(role)) to.srlPredicates.get(role).addAll(fromSrlPredicates);
      else to.srlPredicates.put(role, fromSrlPredicates);
    }
  }

  public boolean isSinglePronounCluster(Dictionaries dict){
    if(this.corefMentions.size() > 1) return false;
    for(Mention m : this.corefMentions) {
      if(m.isPronominal() || dict.allPronouns.contains(m.lowercaseSpan)) return true;
    }
    return false;
  }

  /** Print cluster information */
  public void printCorefCluster(Logger logger, Document doc){
    logger.fine("Cluster ID: "+clusterID+"\tNumbers: "+numbers+"\tGenders: "+genders+"\tanimacies: "+animacies);
    logger.fine("NE: "+nerStrings+"\tfirst Mention's ID: "+firstMention.mentionID+"\tHeads: "+heads+"\twords: "+words);
    logger.fine("SrlArgs: "+srlRoles+"\tSrlPredicates: "+srlPredicates);
    TreeMap<Double, Mention> forSortedPrint = new TreeMap<Double, Mention>();
    for(Mention m : this.corefMentions){
      forSortedPrint.put(m.sentNum+m.mentionID*1.0/10000, m);
    }
    for(Mention m : forSortedPrint.values()){
      String event = (m.isEvent)? "event" : "entity";
      String verb = (m.isVerb)? "verb" : "noun";
      String report = (m.isReport)? "report" : "not report";
      String gold = (m.twinless)? "not gold" : "gold";

      if(m.twinless) {
        logger.fine("mention-> id:"+m.mentionID+"\t"+m.spanToString()+"\t"+gold+"\t"+event+"\t"+report+"\t"+verb+"\tsentNum: "+m.sentNum+"\tstartIndex: "+m.startIndex);
      } else {
        logger.fine("mention-> id:"+m.mentionID+"\t"+m.spanToString()+"\t"+gold+"\tgoldClusterID:"+doc.allGoldMentions.get(m.mentionID).goldCorefClusterID+"\t"+event+"\t"+report+"\t"+verb+"\tsentNum: "+m.sentNum+"\tstartIndex: "+m.startIndex);
      }
    }
  }
  public static boolean nominalCluster(CorefCluster c) {
    for(Mention m : c.corefMentions) {
      if(m.isVerb) return false;
    }
    return true;
  }
}
