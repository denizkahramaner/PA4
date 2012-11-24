package edu.stanford.nlp.jcoref.dcoref;

import java.util.Map;

public class ScorerPairwise extends CorefScorer {

  private final boolean removeSpuriousMentions;
  protected final boolean eventScore;

  public ScorerPairwise(boolean removeSpuriousMentions, boolean eventScore){
    super();
    this.scoreType = ScoreType.Pairwise;
    this.removeSpuriousMentions = removeSpuriousMentions;
    this.eventScore = eventScore;
  }

  @Override
  protected void calculateRecall(Document doc) {
    int rDen = 0;
    int rNum = 0;
    Map<Integer, Mention> predictedMentions = doc.allPredictedMentions;

    for(CorefCluster g : doc.goldCorefClusters.values()) {
      //      int clusterSize = g.getCorefMentions().size();
      //      rDen += clusterSize*(clusterSize-1)/2;
      for(Mention m1 : g.getCorefMentions()){
        for(Mention m2 : g.getCorefMentions()) {
          if(m1.mentionID > m2.mentionID || m1.isEvent!=eventScore || m2.isEvent!=eventScore) continue;
          rDen++;
          if(predictedMentions.containsKey(m1.mentionID) && predictedMentions.containsKey(m2.mentionID)
              && predictedMentions.get(m1.mentionID).corefClusterID == predictedMentions.get(m2.mentionID).corefClusterID){
            rNum++;
          }
        }
      }
    }
    recallDenSum += rDen;
    recallNumSum += rNum;
  }

  @Override
  protected void calculatePrecision(Document doc) {
    int pDen = 0;
    int pNum = 0;

    Map<Integer, Mention> goldMentions = doc.allGoldMentions;

    for(CorefCluster c : doc.corefClusters.values()){
      //      int clusterSize = c.getCorefMentions().size();
      //      pDen += clusterSize*(clusterSize-1)/2;
      for(Mention m1 : c.getCorefMentions()){
        for(Mention m2 : c.getCorefMentions()) {
          if(m1.mentionID > m2.mentionID || m1.isEvent!=eventScore || m2.isEvent!=eventScore) continue;
          boolean bothInGold = goldMentions.containsKey(m1.mentionID) && goldMentions.containsKey(m2.mentionID);
          if(removeSpuriousMentions && !bothInGold) {
            continue;
          }
          pDen++;
          if(bothInGold
              && goldMentions.get(m1.mentionID).goldCorefClusterID == goldMentions.get(m2.mentionID).goldCorefClusterID){
            pNum++;
          }
        }
      }
    }
    precisionDenSum += pDen;
    precisionNumSum += pNum;
  }
}
