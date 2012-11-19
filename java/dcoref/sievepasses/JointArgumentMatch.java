package edu.stanford.nlp.jcoref.dcoref.sievepasses;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import edu.stanford.nlp.jcoref.JointCorefClassifier;
import edu.stanford.nlp.jcoref.RuleBasedJointCorefSystem;
import edu.stanford.nlp.jcoref.dcoref.CorefCluster;
import edu.stanford.nlp.jcoref.dcoref.Dictionaries;
import edu.stanford.nlp.jcoref.dcoref.Dictionaries.Animacy;
import edu.stanford.nlp.jcoref.dcoref.Dictionaries.Gender;
import edu.stanford.nlp.jcoref.dcoref.Dictionaries.MentionType;
import edu.stanford.nlp.jcoref.dcoref.Dictionaries.Number;
import edu.stanford.nlp.jcoref.dcoref.Document;
import edu.stanford.nlp.jcoref.dcoref.Mention;
import edu.stanford.nlp.jcoref.dcoref.Rules;
import edu.stanford.nlp.jcoref.dcoref.ScorerMUC;
import edu.stanford.nlp.jcoref.dcoref.SieveCoreferenceSystem;
import edu.stanford.nlp.jcoref.docclustering.SimilarityVector;
import edu.stanford.nlp.ling.CoreAnnotations.LemmaAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TextAnnotation;
import edu.stanford.nlp.ling.RVFDatum;
import edu.stanford.nlp.stats.ClassicCounter;
import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.stats.Counters;
import edu.stanford.nlp.util.IntPair;

public class JointArgumentMatch extends DeterministicCorefSieve {

  private static final boolean DEBUG = true;

  //  public static Date startTime;
  //  public static Counter<String> time = new ClassicCounter<String>();
  public static final Map<String, Double> featureWeight = new HashMap<String, Double>();
  public static final double CUTOFF_SIMILARITY = 0.4; // don't store similarity for cluster pair if they are not similar than this cutoff.
  
  // TODO : fix later - not thread safe
  public static double CUTOFF_THRESHOLD = 0.5; // don't store similarity for cluster pair if they are not similar than this cutoff.

  public static boolean SRL_INDICATOR = true;
  public static boolean USE_DISAGREE = false;
  public static boolean USE_RULES = true;
  public static boolean DOPRONOUN = true;
  public static Map<IntPair, RVFDatum<Double, String>> rawFeatures = new HashMap<IntPair, RVFDatum<Double, String>> ();

  public JointArgumentMatch() throws IOException, ClassNotFoundException {
    super();
    flags.JOINT_ARG_MATCH = true;
    flags.USE_EVENT_iwithini = true;
    featureWeight.put("LEMMA", 1.0);
    featureWeight.put("HEAD", 1.0);
    featureWeight.put("MENTION_WORDS", 1.0);
    featureWeight.put("SENTENCE_WORDS", 1.0);
    featureWeight.put("SYNONYM", 1.0);
    featureWeight.put("SRLROLES-A0", 1.0);
    featureWeight.put("SRLPREDS-A0", 1.0);
    featureWeight.put("SRLROLES-A1", 1.0);
    featureWeight.put("SRLPREDS-A1", 1.0);
    featureWeight.put("SRLROLES-RIGHT-MENTION", 1.0);
  }
  public JointArgumentMatch(String args) throws IOException, ClassNotFoundException {
    this();
    flags.set(args);
    for(String arg : args.split(",")) {
      if(arg.startsWith("CLASSIFIER_CUTOFF:")) {
        this.CUTOFF_THRESHOLD = Double.parseDouble(arg.split("CLASSIFIER_CUTOFF:")[1]);
        RuleBasedJointCorefSystem.logger.fine("JointArgumentMatch classifier cutoff: "+CUTOFF_THRESHOLD);
      }
    }
  }
//  public boolean jointArgCoref(Document document, Dictionaries dict, JointCorefClassifier jcc, boolean trainJcc) throws IOException {
//    if(trainJcc) return jointArgCoref(document, dict, jcc);
//    else {
//      
//    }
//  }
  public boolean jointArgCoref(Document document, Dictionaries dict, JointCorefClassifier jcc) throws IOException {
    boolean changed = false;

    // best clusters to merge
    IntPair clustersToMerge;
    double previousScore = 1;

    // merge two clusters
    while((clustersToMerge=getClustersToMerge(document)) != null) {
      // check if both clusters are not removed in previous loop.
      if(!document.corefClusters.containsKey(clustersToMerge.get(0))
          || !document.corefClusters.containsKey(clustersToMerge.get(1))) {
        document.corefScore.remove(clustersToMerge);
        continue;
      }

      int mergedID = clustersToMerge.get(0);
      int removeID = clustersToMerge.get(1);
      CorefCluster cMerged = document.corefClusters.get(mergedID);
      CorefCluster cRemove = document.corefClusters.get(removeID);

      // rules here
      boolean skipThis = false;
      LOOP:
        if(cMerged.getRepresentativeMention().isVerb) {
        // check event i-within-i
          if(Rules.eventIWithinI(cMerged, cRemove, document)) {
            document.corefScore.remove(clustersToMerge);
            continue;
          }
          if(USE_RULES) {
            for(Mention m1 : cMerged.getCorefMentions()) {
              for(Mention m2 : cRemove.getCorefMentions()) {
  //              if(m1.sentNum==m2.sentNum) {
  //                skipThis = true;
  //                break LOOP;
  //              }
                for(String role1 : m1.srlArgs.keySet()) {
                  for(String role2 : m2.srlArgs.keySet()) {
                    if(role1.contains("LEFT") || role1.contains("RIGHT") || role2.contains("LEFT") || role2.contains("RIGHT")) continue;
                    if(!role1.equals(role2) && m1.srlArgs.get(role1)!=null && m2.srlArgs.get(role2)!=null 
                        && m1.srlArgs.get(role1).corefClusterID==m2.srlArgs.get(role2).corefClusterID) {
                      skipThis = true;
                      break LOOP;
                    }
                  }
                }
              }
            }
          }
        } else {
        // check entity i-within-i
          if(Rules.entityIWithinI(cMerged, cRemove, dict)) {
            document.corefScore.remove(clustersToMerge);
            continue;
          }
          if(USE_RULES) {
            if(Rules.entityHaveDifferentNumberMention(cMerged.getRepresentativeMention(), cRemove.getRepresentativeMention())
                || Rules.entityHaveDifferentLocation(cMerged.getRepresentativeMention(), cRemove.getRepresentativeMention(), dict)) {
              skipThis = true;
              break LOOP;
            }
            for(Mention m1 : cMerged.getCorefMentions()) {
              for(Mention m2 : cRemove.getCorefMentions()) {
  //              if(m1.sentNum==m2.sentNum) {
  //                skipThis = true;
  //                break LOOP;
  //              }
                for(Mention m1Pred : m1.srlPredicates.keySet()) {
                  for(Mention m2Pred : m2.srlPredicates.keySet()) {
                    String role1 = m1.srlPredicates.get(m1Pred);
                    String role2 = m2.srlPredicates.get(m2Pred);
                    if(role1.contains("LEFT") || role1.contains("RIGHT") || role2.contains("LEFT") || role2.contains("RIGHT")) continue;
                    if(m1Pred.corefClusterID==m2.corefClusterID && !role1.equals(role2)) {
                      skipThis = true;
                      break LOOP;
                    }
                  }
                }
              }
            }
          }
        }
      if(skipThis) {
        document.corefScore.remove(clustersToMerge);
        continue;
      }


      // end of rules

      if(DEBUG) {
        logClusterMerge(document, cMerged, cRemove, clustersToMerge);
      }
      
      CorefCluster.mergeClusters(cMerged, cRemove);
      changed = true;
      document.corefClusters.remove(removeID);
      
      double score = document.corefScore.getCount(clustersToMerge);
      boolean doRecalculation = (score < previousScore*0.8);
      
      if(SieveCoreferenceSystem.trainJointCorefClassifier || !doRecalculation) {
        JointArgumentMatch.calculateCentroid(cMerged, document, false);
        JointArgumentMatch.calculateCentroid(cMerged, document, true);
        for(Integer clusterID : document.corefClusters.keySet()) {
          CorefCluster c2 = document.corefClusters.get(clusterID);
          if(mergedID == clusterID || (!JointArgumentMatch.DOPRONOUN && (cMerged.getRepresentativeMention().isPronominal() || c2.getRepresentativeMention().isPronominal()))) continue;
//              ) continue;
          IntPair idPair = (mergedID < clusterID)? new IntPair(mergedID, clusterID) : new IntPair(clusterID, mergedID);

          double sim;
          Counter<String> features = null;
          if(jcc.regressor==null) {
            sim = JointArgumentMatch.calculateSimilarity(document, cMerged, c2);
          } else {
            features = getFeatures(document, cMerged, c2, false, dict);
            RVFDatum<Double, String> datum = new RVFDatum<Double, String>(features);
            JointArgumentMatch.rawFeatures.put(idPair, datum);
            sim = jcc.valueOf(datum);
          }

          if(sim > CUTOFF_THRESHOLD) {
            if((!cMerged.getRepresentativeMention().isPronominal() && !c2.getRepresentativeMention().isPronominal()) || (features!=null && features.getCount("SRLAGREECOUNT")!=0)) {
              document.corefScore.setCount(idPair, sim);
            }
          }
          if(SieveCoreferenceSystem.trainJointCorefClassifier) {
            double s = getMergeScore(document, cMerged, c2);
            if(Math.abs(s-0.5) > 0.0001){
              jcc.addData(new RVFDatum<Double, String>(getFeatures(document, cMerged, c2, true, dict), s));
            }
          }
        }
      } else {
        previousScore = score;
        document.initializeArgMatchCounter(jcc, dict, true);
      }
//      if(document.mergingList == null) document.mergingList = new ArrayList<IntPair>();
//      document.mergingList.add(new IntPair(mergedID, removeID));

      // don't need calinski measure when we know merging is good or bad from classifier
//      if(document.calinskiScore == null) document.calinskiScore = new ClassicCounter<Integer>();
//      calculateCalinskiScore(document);
//      RuleBasedJointCorefSystem.logger.fine("calinski score: "+document.corefClusters.size() + " -> "+document.calinskiScore.getCount(document.corefClusters.size()));
    }
//    for(int size : document.calinskiScore.keySet()) {
//      System.err.println(size + " :: "+document.calinskiScore.getCount(size));
//    }
    //    System.out.println(time);
    return changed;
  }
  public static double getMergeScore(Document document, CorefCluster c1, CorefCluster c2) {
    double correct = 0.01;    // prevent dividing by 0
    double incorrect = 0.01;
    for(Mention m1 : c1.getCorefMentions()) {
      for(Mention m2 : c2.getCorefMentions()) {
        if(document.allGoldMentions.containsKey(m1.mentionID) && document.allGoldMentions.containsKey(m2.mentionID)) {
          if(document.allGoldMentions.get(m1.mentionID).goldCorefClusterID == document.allGoldMentions.get(m2.mentionID).goldCorefClusterID) {
            correct++;
          } else {
            incorrect++;
          }
        }
      }
    }
    return correct/(correct+incorrect);
  }
  private double calculateNewMUCF1(Document document, CorefCluster c1, CorefCluster c2, double oldF1) {
    ScorerMUC s = new ScorerMUC();
    s.calculateScore(document, c1, c2);
    double f1 = s.getF1();
    return s.getF1();
  }
  private double calculateMUCF1(Document document) throws IOException {
    ScorerMUC s = new ScorerMUC();
    s.calculateScore(document, true);
//    double f1 = s.getF1();
    return s.getF1();
  }
  public static void logClusterMerge(Document document, CorefCluster cMerged, CorefCluster cRemove, IntPair clustersToMerge) {

    int correctEntityLinksCount = 0;
    int correctEventLinksCount = 0;
    int incorrectEntityLinksCount = 0;
    int incorrectEventLinksCount = 0;
    for(Mention m : cMerged.getCorefMentions()) {
      for(Mention a : cRemove.getCorefMentions()) {
        if(document.allGoldMentions.containsKey(m.mentionID) && document.allGoldMentions.containsKey(a.mentionID)) {
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
        } else if(!document.allGoldMentions.containsKey(m.mentionID) && !document.allGoldMentions.containsKey(a.mentionID)) {
          if(document.goldOrderedMentionsBySentence.get(a.sentNum).size() > 0 && document.goldOrderedMentionsBySentence.get(m.sentNum).size() > 0){
            if(m.isEvent) incorrectEventLinksCount++;
            else incorrectEntityLinksCount++;
          }
        } else {
          if(document.allGoldMentions.containsKey(m.mentionID) && document.goldOrderedMentionsBySentence.get(a.sentNum).size() > 0) {
            if(m.isEvent) incorrectEventLinksCount++;
            else incorrectEntityLinksCount++;
          } else if(document.allGoldMentions.containsKey(a.mentionID) && document.goldOrderedMentionsBySentence.get(m.sentNum).size() > 0) {
            if(a.isEvent) incorrectEventLinksCount++;
            else incorrectEntityLinksCount++;
          }
        }
      }
    }
    double sim = document.corefScore.getCount(clustersToMerge);

    RuleBasedJointCorefSystem.correctEntityLinksFoundForThreshold.incrementCount(Math.floor(sim), correctEntityLinksCount);
    RuleBasedJointCorefSystem.incorrectEntityLinksFoundForThreshold.incrementCount(Math.floor(sim), incorrectEntityLinksCount);
    RuleBasedJointCorefSystem.correctEventLinksFoundForThreshold.incrementCount(Math.floor(sim), correctEventLinksCount);
    RuleBasedJointCorefSystem.incorrectEventLinksFoundForThreshold.incrementCount(Math.floor(sim), incorrectEventLinksCount);

    // print merging & the number of correct/incorrect links
    RuleBasedJointCorefSystem.logger.fine("\nJoint Argument Match -----------------");
    RuleBasedJointCorefSystem.logger.fine("similarity: "+sim);
    RuleBasedJointCorefSystem.logger.fine("features: "+rawFeatures.get(clustersToMerge));
    RuleBasedJointCorefSystem.logger.fine("entity links (correct/incorrect): "+correctEntityLinksCount+" / "+incorrectEntityLinksCount);
    RuleBasedJointCorefSystem.logger.fine("event links (correct/incorrect): "+correctEventLinksCount+" / "+incorrectEventLinksCount);
    if(rawFeatures.containsKey(clustersToMerge)) {
      boolean noLemmaMatch = (rawFeatures.get(clustersToMerge).asFeaturesCounter().getCount("LEMMA") == 0);
      if(cMerged.getRepresentativeMention().isVerb) RuleBasedJointCorefSystem.logger.fine("Good event merge: "+(correctEventLinksCount>incorrectEventLinksCount)+", no lemma match: "+noLemmaMatch);
    }
    for(String role : cMerged.srlPredicates.keySet()) {
      if(!cRemove.srlPredicates.containsKey(role)) continue;
      for(Mention m1 : cMerged.srlPredicates.get(role)) {
        for(Mention m2 : cRemove.srlPredicates.get(role)) {
          if(m1!=null && m2!=null && m1.corefClusterID==m2.corefClusterID) {
            RuleBasedJointCorefSystem.logger.fine("Coreferent SRL Predicates: "+role+" - "+m1.spanToString());
          }
        }
      }
    }
    for(String role : cMerged.srlRoles.keySet()) {
      if(!cRemove.srlRoles.containsKey(role)) continue;
      for(Mention m1 : cMerged.srlRoles.get(role)) {
        for(Mention m2 : cRemove.srlRoles.get(role)) {
          if(m1!=null && m2!=null && m1.corefClusterID==m2.corefClusterID) {
            RuleBasedJointCorefSystem.logger.fine("Coreferent SRL Roles: "+role+" - "+m1.spanToString());
          }
        }
      }
    }

    cMerged.printCorefCluster(RuleBasedJointCorefSystem.logger, document);
    RuleBasedJointCorefSystem.logger.fine("--------------------");
    cRemove.printCorefCluster(RuleBasedJointCorefSystem.logger, document);
    RuleBasedJointCorefSystem.logger.fine("---------------------------------------------------------");
  }
  private void calculateCalinskiScore(Document document) {
    double B = 0.0; // between cluster dist
    double W = 0.0; // within cluster dist

    // calculate meta centroid
    calculateMetaCentroid(document);

    // calculate centroids
    for(CorefCluster c : document.corefClusters.values()) {
      calculateCentroid(c, document, false);
    }

    // calculate B
    for(CorefCluster c : document.corefClusters.values()) {
      B += calculateBetweenClusterDistance(c, document);
    }

    // calculate W
    for(Mention m : document.allPredictedMentions.values()) {
      double sim = calculateSimilarity(document, m, document.corefClusters.get(m.corefClusterID));
      //      W += 1 / (sim*sim);
      W += (1-sim)*(1-sim);
    }

    int k = document.corefClusters.size();
    // TODO (n-k)^2, (w(k-1))^2?
    double calinskiScore = B*(document.allPredictedMentions.size() - k) / (W*(k-1));
    document.calinskiScore.setCount(k, calinskiScore);
  }

  private static double calculateBetweenClusterDistance(CorefCluster c, Document doc) {
    double sim = 0.0;
    double simSynonym = 0.0;
    double simSentWords = 0.0;
    double simOthers = 0.0;

    int sameSentCount = 0;
    for(Mention m1 : c.getCorefMentions()) {
      for(Mention m2 : doc.allPredictedMentions.values()) {
        if(m1.isVerb!=m2.isVerb) continue;
        if(m1.mentionID==m2.mentionID) {
          simSynonym++;
          simSentWords++;
          continue;
        }
        simSynonym += (doc.mentionSynonymInWN.contains(new IntPair(Math.min(m1.mentionID, m2.mentionID), Math.max(m1.mentionID, m2.mentionID))))? 1 : 0;
        if(m1.sentNum == m2.sentNum) {    // avoid to have perfect sentence similarity between mentions in the same sentence
          sameSentCount++;
          continue;
        } else {
          simSentWords += doc.sent1stOrderSimilarity.get(new IntPair(Math.min(m1.sentNum, m2.sentNum), Math.max(m1.sentNum, m2.sentNum)));
        }
      }
    }

    int mentionPairCounts = c.getCorefMentions().size()*doc.allPredictedMentions.size();

    simSynonym /= mentionPairCounts;
    simSentWords /= (mentionPairCounts - sameSentCount);

    HashMap<String, ClassicCounter<String>> metaCentroid = (c.getRepresentativeMention().isVerb)? doc.verbMetaCentroid : doc.nominalMetaCentroid;

    double weightSum = featureWeight.get("SYNONYM")+featureWeight.get("SENTENCE_WORDS");

    for(String feature : c.predictedCentroid.keySet()) {
      if(!metaCentroid.containsKey(feature)) {
        continue;
      }
      Counter<String> centFeature1 = c.predictedCentroid.get(feature);
      Counter<String> centFeature2 = metaCentroid.get(feature);

      if(feature.equals("LEMMA") && centFeature1.getCount("say")>0 && centFeature2.getCount("say") > 0) continue;

      double weight = (featureWeight.containsKey(feature))? featureWeight.get(feature) : 1.0;
      weightSum += weight;
      simOthers += weight*SimilarityVector.getCosineSimilarity(new SimilarityVector(centFeature1), new SimilarityVector(centFeature2));
    }

    sim = simOthers + featureWeight.get("SYNONYM")*simSynonym + featureWeight.get("SENTENCE_WORDS")*simSentWords;
    sim /= weightSum;

    //    return c.getCorefMentions().size() / (sim*sim);
    return c.getCorefMentions().size() * (1-sim)*(1-sim);
  }
  public static double calculateSimilarity(Document document, Mention m, CorefCluster c){
    double sim = 0.0;
    double simSynonym = 0.0;
    double simSentWords = 0.0;
    //    double simOthers = calculateSimilarity(m, c.centroid, document);
    double simOthers = 0.0;

    int sameSentCount = 0;
    int mentionPairCounts = c.getCorefMentions().size();
    for(Mention mention : c.getCorefMentions()) {
      if(mention==m) {
        simSynonym++;
        simSentWords++;
        continue;
      }
      simSynonym += (document.mentionSynonymInWN.contains(new IntPair(Math.min(mention.mentionID, m.mentionID), Math.max(mention.mentionID, m.mentionID))))? 1 : 0;
      if(mention.sentNum == m.sentNum) {
        sameSentCount++;
        continue;
      } else {
        simSentWords += document.sent1stOrderSimilarity.get(new IntPair(Math.min(mention.sentNum, m.sentNum), Math.max(mention.sentNum, m.sentNum)));
      }
    }

    simSynonym /= mentionPairCounts;
    if(mentionPairCounts!=sameSentCount) simSentWords /= (mentionPairCounts - sameSentCount);

    double weightSum = featureWeight.get("SYNONYM")+featureWeight.get("SENTENCE_WORDS");
    HashMap<String, ClassicCounter<String>> mentionVector = new HashMap<String, ClassicCounter<String>>();
    addMentionToCentroid(mentionVector, m, document, false);

    for(String feature : mentionVector.keySet()) {
      if(!c.predictedCentroid.containsKey(feature)) {
        continue;
      }
      Counter<String> centFeature1 = c.predictedCentroid.get(feature);
      Counter<String> centFeature2 = mentionVector.get(feature);

      if(feature.equals("LEMMA") && centFeature1.getCount("say")>0 && centFeature2.getCount("say") > 0) continue;

      double weight = (featureWeight.containsKey(feature))? featureWeight.get(feature) : 1.0;
      weightSum += weight;
      simOthers += weight*SimilarityVector.getCosineSimilarity(new SimilarityVector(centFeature1), new SimilarityVector(centFeature2));
    }

    sim = simOthers + featureWeight.get("SYNONYM")*simSynonym + featureWeight.get("SENTENCE_WORDS")*simSentWords;
    sim /= weightSum;
    return sim;
  }

  public static Counter<String> getFeatures(Document document, CorefCluster c1, CorefCluster c2, boolean gold, Dictionaries dict){

    CorefCluster former;
    CorefCluster latter;
    HashMap<String, ClassicCounter<String>> formerCentroid;
    HashMap<String, ClassicCounter<String>> latterCentroid;
    
    if(c1.getRepresentativeMention().appearEarlierThan(c2.getRepresentativeMention())) {  // c1 comes first
      former = c1;
      latter = c2;
      if(gold) {
        formerCentroid = c1.goldCentroid;
        latterCentroid = c2.goldCentroid;
      } else {
        formerCentroid = c1.predictedCentroid;
        latterCentroid = c2.predictedCentroid;
      }
    } else {  // c2 comes first
      former = c2;
      latter = c1;
      if(gold) {
        formerCentroid = c2.goldCentroid;
        latterCentroid = c1.goldCentroid;
      } else {
        formerCentroid = c2.predictedCentroid;
        latterCentroid = c1.predictedCentroid;
      }
    }
    Mention formerRep = former.getRepresentativeMention();
    Mention latterRep = latter.getRepresentativeMention();
    boolean isVerb = latterRep.isVerb || formerRep.isVerb;

    String mentionType = ""; 
    if(!isVerb) {
      if(formerRep.mentionType==MentionType.PROPER && latterRep.mentionType==MentionType.PROPER) mentionType = "-PROPER";
      else if(formerRep.mentionType==MentionType.PRONOMINAL || latterRep.mentionType==MentionType.PRONOMINAL) mentionType = "-PRONOMINAL";
      else mentionType = "-NOMINAL";
    }
    Counter<String> features = new ClassicCounter<String>();
    
    // TODO : temp for debug
//    if(mentionType.equals("-PRONOMINAL")) return features;

    double headNom = 0.0;
    double headDenom = 0.0;
    double synonymNom = 0.0;
    double synonymDenom = 0.0;

    for(Mention m1 : c1.getCorefMentions()) {
      for(Mention m2 : c2.getCorefMentions()) {
        
        if(!JointArgumentMatch.DOPRONOUN && (m1.isPronominal() || m2.isPronominal())) continue;
        IntPair menPair = new IntPair(Math.min(m1.mentionID, m2.mentionID), Math.max(m1.mentionID, m2.mentionID));

        if(isVerb) {
          synonymDenom++;
          if(document.mentionSynonymInWN.contains(menPair)) {
            synonymNom++;
          }
        } else {
          synonymDenom++;
          if(document.mentionSynonymInWN.contains(menPair)) {
            synonymNom++;
          }
        }
      }
    }

    if(isVerb) {
      features.incrementCount("SYNONYM", synonymNom/synonymDenom);
    } else {
      if(synonymDenom > 0) {
        if(!mentionType.equals("-PRONOMINAL")) features.incrementCount("SYNONYM"+mentionType, synonymNom/synonymDenom);
      }
    }

    for(String feature : formerCentroid.keySet()) {
      if(!latterCentroid.containsKey(feature)) {
        continue;
      }
      Counter<String> centFeature1 = latterCentroid.get(feature);
      Counter<String> centFeature2 = formerCentroid.get(feature);

      if(mentionType.equals("-PRONOMINAL") && (feature.startsWith("MENTION_WORD") || feature.startsWith("HEAD"))) continue;
      if(feature.equals("LEMMA") && centFeature1.getCount("say")>0 && centFeature2.getCount("say") > 0) continue;
      if(feature.startsWith("SRL")) {
        Set<String> featureSet1 = new HashSet<String>();
        featureSet1.addAll(centFeature1.keySet());
        featureSet1.retainAll(centFeature2.keySet());
        int overlap = 0;
        for(String f : featureSet1){
          overlap += centFeature1.getCount(f)* centFeature2.getCount(f);
        }
        if(SRL_INDICATOR) {
          if(featureSet1.size() > 0) {
            features.incrementCount(feature+mentionType);
          }
        } else {
          features.incrementCount(feature+mentionType, overlap);
        }
      } else {
        features.incrementCount(feature+mentionType, SimilarityVector.getCosineSimilarity(new SimilarityVector(centFeature1), new SimilarityVector(centFeature2)));
      }
      if(USE_DISAGREE && feature.startsWith("SRL")) features.incrementCount("DISAGREE"+feature+mentionType, 1-SimilarityVector.getCosineSimilarity(new SimilarityVector(centFeature1), new SimilarityVector(centFeature2)));
    }
    if(USE_DISAGREE && features.containsKey("NUMBER"+mentionType)) features.incrementCount("NUMBER_DISAGREE"+mentionType, 1-features.getCount("NUMBER"+mentionType));
    if(USE_DISAGREE && features.containsKey("GENDER"+mentionType)) features.incrementCount("GENDER_DISAGREE"+mentionType, 1-features.getCount("GENDER"+mentionType));
    if(USE_DISAGREE && features.containsKey("ANIMACY"+mentionType)) features.incrementCount("ANIMACY_DISAGREE"+mentionType, 1-features.getCount("ANIMACY"+mentionType));

    boolean noLeft = false;
    boolean noRight = false;
    String left = "";
    String right = "";
    int srlAgreeCount = 0;
    for(String feature : features.keySet()) {
      if(feature.startsWith("SRL")) {
        if(features.getCount(feature) > 0) srlAgreeCount++;
      }
      if(isVerb) {
        if(feature.startsWith("SRLROLES-A0")) noLeft = true;
        if(feature.startsWith("SRLROLES-A1") || feature.startsWith("SRLPRED-A0")) noRight = true;
      }
      if(feature.contains("LEFT")) left = feature;
      if(feature.contains("RIGHT")) right = feature;
    }
    features.incrementCount("SRLAGREECOUNT", srlAgreeCount);
    if(noLeft) features.remove(left);
    if(noRight) features.remove(right);

    if(features.containsKey("HEAD")){
      features.setCount("HEAD-NOMINAL", features.getCount("HEAD"));
      features.remove("HEAD");
    }
    if(features.containsKey("NUMBER")){
      features.setCount("NUMBER-NOMINAL", features.getCount("NUMBER"));
      features.remove("NUMBER");
    }
    if(features.containsKey("GENDER")){
      features.setCount("GENDER-NOMINAL", features.getCount("GENDER"));
      features.remove("GENDER");
    }
    if(features.containsKey("ANIMACY")){
      features.setCount("ANIMACY-NOMINAL", features.getCount("ANIMACY"));
      features.remove("ANIMACY");
    }
    if(features.containsKey("NETYPE")){
      features.setCount("NETYPE-NOMINAL", features.getCount("NETYPE"));
      features.remove("NETYPE");
    }
    if(features.containsKey("MENTION_WORDS")){
      features.setCount("MENTION_WORDS-NOMINAL", features.getCount("MENTION_WORDS"));
      features.remove("MENTION_WORDS");
    }
    return features;
  }
  public static double calculateSimilarity(Document document, CorefCluster c1, CorefCluster c2){
    double sim = 0.0;
    double simSynonym = 0.0;
    double simSentWords = 0.0;

    int sameSentCount = 0;
    for(Mention m1 : c1.getCorefMentions()) {
      for(Mention m2 : c2.getCorefMentions()) {
        IntPair menPair = new IntPair(Math.min(m1.mentionID, m2.mentionID), Math.max(m1.mentionID, m2.mentionID));
        simSynonym += (document.mentionSynonymInWN.contains(menPair))? 1 : 0;
        if(m1.sentNum == m2.sentNum) {
          sameSentCount++;
          continue;
        } else {
          simSentWords += document.sent1stOrderSimilarity.get(new IntPair(Math.min(m1.sentNum, m2.sentNum), Math.max(m1.sentNum, m2.sentNum)));
        }
      }
    }

    int mentionPairCounts = c1.getCorefMentions().size()*c2.getCorefMentions().size();

    simSynonym /= mentionPairCounts;
    if(mentionPairCounts - sameSentCount != 0) simSentWords /= (mentionPairCounts - sameSentCount);

    double simOthers = 0.0;
    double weightSum = featureWeight.get("SYNONYM")+featureWeight.get("SENTENCE_WORDS");

    for(String feature : c2.predictedCentroid.keySet()) {
      if(!c1.predictedCentroid.containsKey(feature)) {
        continue;
      }
      Counter<String> centFeature1 = c1.predictedCentroid.get(feature);
      Counter<String> centFeature2 = c2.predictedCentroid.get(feature);

      if(feature.equals("LEMMA") && centFeature1.getCount("say")>0 && centFeature2.getCount("say") > 0) continue;

      double weight = (featureWeight.containsKey(feature))? featureWeight.get(feature) : 1.0;
      weightSum += weight;
      simOthers += weight*SimilarityVector.getCosineSimilarity(new SimilarityVector(centFeature1), new SimilarityVector(centFeature2));
    }

    sim = simOthers + featureWeight.get("SYNONYM")*simSynonym + featureWeight.get("SENTENCE_WORDS")*simSentWords;
    sim /= weightSum;
    return sim;
  }

  private static void calculateMetaCentroid(Document document) {
    document.verbMetaCentroid = new HashMap<String, ClassicCounter<String>>();
    document.nominalMetaCentroid = new HashMap<String, ClassicCounter<String>>();

    for(Mention m : document.allPredictedMentions.values()) {
      if(m.isVerb) addMentionToCentroid(document.verbMetaCentroid, m, document, false);
      else addMentionToCentroid(document.nominalMetaCentroid, m, document, false);
    }
  }
  public static void calculateCentroid(CorefCluster c, Document document, boolean gold) {
    if(gold) {
      c.goldCentroid = new HashMap<String, ClassicCounter<String>>();
      for(Mention m : c.getCorefMentions()) {
        addMentionToCentroid(c.goldCentroid, m, document, gold);
      }
    } else {
      c.predictedCentroid = new HashMap<String, ClassicCounter<String>>();
      for(Mention m : c.getCorefMentions()) {
        addMentionToCentroid(c.predictedCentroid, m, document, gold);
      }
    }
  }
  private static void addMentionToCentroid(
      HashMap<String, ClassicCounter<String>> centroid, Mention m, Document doc, boolean gold) {
    // synonym and sentence words are compared in pairwise
    if(!centroid.containsKey("LEMMA")) centroid.put("LEMMA", new ClassicCounter<String>());
    centroid.get("LEMMA").incrementCount(m.headWord.get(LemmaAnnotation.class));
    if(m.isVerb) {
      for(String role : m.srlArgs.keySet()) {
        if(m.srlArgs.get(role)==null) continue;
        if(!centroid.containsKey("SRLROLES-"+role)) centroid.put("SRLROLES-"+role, new ClassicCounter<String>());
        if(gold) {
          if(doc.allGoldMentions.containsKey(m.srlArgs.get(role).mentionID)) {
            int goldID = doc.allGoldMentions.get(m.srlArgs.get(role).mentionID).goldCorefClusterID;
            centroid.get("SRLROLES-"+role).incrementCount(Integer.toString(goldID));
          }
        } else {
          centroid.get("SRLROLES-"+role).incrementCount(Integer.toString(m.srlArgs.get(role).corefClusterID));
        }
      }
    } else {
      if(!centroid.containsKey("GENDER") && m.gender != Gender.UNKNOWN) centroid.put("GENDER", new ClassicCounter<String>());
      if(!centroid.containsKey("NUMBER") && m.number != Number.UNKNOWN) centroid.put("NUMBER", new ClassicCounter<String>());
      if(!centroid.containsKey("ANIMACY")&& m.animacy!= Animacy.UNKNOWN) centroid.put("ANIMACY", new ClassicCounter<String>());
      if(!centroid.containsKey("NETYPE") && !m.nerString.equals("O")) centroid.put("NETYPE", new ClassicCounter<String>());
      if(!centroid.containsKey("MENTION_WORDS")) centroid.put("MENTION_WORDS", new ClassicCounter<String>());
      if(!centroid.containsKey("HEAD") && !m.isPronominal()) centroid.put("HEAD", new ClassicCounter<String>());
      
      for(String role : m.srlArgs.keySet()) {
        if(m.srlArgs.get(role)==null) continue;
        if(!centroid.containsKey("SRLROLES-"+role)) centroid.put("SRLROLES-"+role, new ClassicCounter<String>());
        if(gold) {
          if(doc.allGoldMentions.containsKey(m.srlArgs.get(role).mentionID)) {
            int goldID = doc.allGoldMentions.get(m.srlArgs.get(role).mentionID).goldCorefClusterID;
            centroid.get("SRLROLES-"+role).incrementCount(Integer.toString(goldID));
          }
        } else {
          centroid.get("SRLROLES-"+role).incrementCount(Integer.toString(m.srlArgs.get(role).corefClusterID));
        }
      }
      for(Mention mention : m.srlPredicates.keySet()) {
        String role = m.srlPredicates.get(mention);
        if(!centroid.containsKey("SRLPREDS-"+role)) centroid.put("SRLPREDS-"+role, new ClassicCounter<String>());
        if(gold) {
          if(doc.allGoldMentions.containsKey(mention.mentionID)) {
            int goldID = doc.allGoldMentions.get(mention.mentionID).goldCorefClusterID;
            centroid.get("SRLPREDS-"+role).incrementCount(Integer.toString(goldID));
          }
        } else {
          centroid.get("SRLPREDS-"+role).incrementCount(Integer.toString(mention.corefClusterID));
        }
      }

      if(!m.isPronominal()) {
        centroid.get("HEAD").incrementCount(m.headWord.get(TextAnnotation.class));
      }
      if(m.gender != Gender.UNKNOWN) centroid.get("GENDER").incrementCount(m.gender.toString());
      if(m.number != Number.UNKNOWN) centroid.get("NUMBER").incrementCount(m.number.toString());
      if(m.animacy!= Animacy.UNKNOWN) centroid.get("ANIMACY").incrementCount(m.animacy.toString());
      if(!m.nerString.equals("O")) centroid.get("NETYPE").incrementCount(m.nerString);

      Counters.addInPlace(centroid.get("MENTION_WORDS"), m.simVector.vector);   // 2nd order
    }
  }

  private static IntPair getClustersToMerge(Document document) {
    return Counters.argmax(document.corefScore);
  }
}
