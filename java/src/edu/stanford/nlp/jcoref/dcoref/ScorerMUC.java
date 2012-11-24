package edu.stanford.nlp.jcoref.dcoref;

import java.util.HashSet;
import java.util.Map;
import java.util.Set;

public class ScorerMUC extends CorefScorer {

  public ScorerMUC(){
    super();
    scoreType = ScoreType.MUC;
  }

  @Override
  protected void calculateRecall(Document doc) {
    int rDen = 0;
    int rNum = 0;

    Map<Integer, Mention> predictedMentions = doc.allPredictedMentions;
    for(CorefCluster g : doc.goldCorefClusters.values()){
      if(g.corefMentions.size()==0) continue;
      rDen += g.corefMentions.size()-1;
      rNum += g.corefMentions.size();

      Set<CorefCluster> partitions = new HashSet<CorefCluster>();
      for (Mention goldMention : g.corefMentions){
        if(!predictedMentions.containsKey(goldMention.mentionID)) {  // twinless goldmention
          rNum--;
        } else {
          partitions.add(doc.corefClusters.get(predictedMentions.get(goldMention.mentionID).corefClusterID));
        }
      }
      rNum -= partitions.size();
    }
    assert(rDen == (doc.allGoldMentions.size()-doc.goldCorefClusters.values().size()));

    recallNumSum += rNum;
    recallDenSum += rDen;
  }

  @Override
  protected void calculatePrecision(Document doc) {
    int pDen = 0;
    int pNum = 0;
    Map<Integer, Mention> goldMentions = doc.allGoldMentions;

    for(CorefCluster c : doc.corefClusters.values()){
      if(c.corefMentions.size()==0) continue;
      pDen += c.corefMentions.size()-1;
      pNum += c.corefMentions.size();
      Set<CorefCluster> partitions = new HashSet<CorefCluster>();
      for (Mention predictedMention : c.corefMentions){
        if(!goldMentions.containsKey(predictedMention.mentionID)) {  // twinless goldmention
          pNum--;
        } else {
          partitions.add(doc.goldCorefClusters.get(goldMentions.get(predictedMention.mentionID).goldCorefClusterID));
        }
      }
      pNum -= partitions.size();
    }
    assert(pDen == (doc.allPredictedMentions.size()-doc.corefClusters.values().size()));

    precisionDenSum += pDen;
    precisionNumSum += pNum;
  }
  public void calculateScore(Document doc, boolean forJointArgumentMatchScoring){
    calculatePrecision(doc, forJointArgumentMatchScoring);    // handling spurious mentions in unannotated sentences
    calculateRecall(doc);
  }
  private void calculatePrecision(Document doc, boolean forJointArgumentMatchScoring) {
    if(forJointArgumentMatchScoring) {
      int pDen = 0;
      int pNum = 0;
      Map<Integer, Mention> goldMentions = doc.allGoldMentions;

      for(CorefCluster c : doc.corefClusters.values()){
        if(c.corefMentions.size()==0) continue;
        pNum += c.corefMentions.size();
        int unannotatedSpurious = 0;
        Set<CorefCluster> partitions = new HashSet<CorefCluster>();
        for (Mention predictedMention : c.corefMentions){
          if(!goldMentions.containsKey(predictedMention.mentionID)) {  // twinless predictedmention
            pNum--;
            if(doc.goldOrderedMentionsBySentence.get(predictedMention.sentNum).size()==0) {
              unannotatedSpurious++;
            }
          } else {
            partitions.add(doc.goldCorefClusters.get(goldMentions.get(predictedMention.mentionID).goldCorefClusterID));
          }
        }
        pNum -= partitions.size();
        pDen += Math.max(0, c.corefMentions.size()-unannotatedSpurious-1);
      }
      assert(pDen == (doc.allPredictedMentions.size()-doc.corefClusters.values().size()));

      precisionDenSum += pDen;
      precisionNumSum += pNum;
    } else {
      calculatePrecision(doc);
    }
  }

  // calculate score assuming c1 and c2 are merged.
  public void calculateScore(Document document, CorefCluster c1, CorefCluster c2) {
    calculatePrecision(document, c1, c2);
    calculateRecall(document, c1, c2);
  }

  private void calculateRecall(Document doc, CorefCluster c1, CorefCluster c2) {
    int rDen = 0;
    int rNum = 0;

    Map<Integer, Mention> predictedMentions = doc.allPredictedMentions;
    for(CorefCluster g : doc.goldCorefClusters.values()){
      if(g.corefMentions.size()==0) continue;
      rDen += g.corefMentions.size()-1;
      rNum += g.corefMentions.size();

      Set<CorefCluster> partitions = new HashSet<CorefCluster>();
      for (Mention goldMention : g.corefMentions){
        if(!predictedMentions.containsKey(goldMention.mentionID)) {  // twinless goldmention
          rNum--;
        } else {
          partitions.add(doc.corefClusters.get(predictedMentions.get(goldMention.mentionID).corefClusterID));
        }
      }
      rNum -= partitions.size();
      if(partitions.contains(c1) && partitions.contains(c2)) rNum++;
    }
    recallNumSum += rNum;
    recallDenSum += rDen;
  }

  private void calculatePrecision(Document doc, CorefCluster c1, CorefCluster c2) {
    int pDen = 0;
    int pNum = 0;
    Map<Integer, Mention> goldMentions = doc.allGoldMentions;

    for(CorefCluster c : doc.corefClusters.values()){
      if(c.corefMentions.size()==0 || c==c1 || c==c2) continue;
      pNum += c.corefMentions.size();
      int unannotatedSpurious = 0;
      Set<CorefCluster> partitions = new HashSet<CorefCluster>();
      for (Mention predictedMention : c.corefMentions){
        if(!goldMentions.containsKey(predictedMention.mentionID)) {  // twinless predictedmention
          pNum--;
          if(doc.goldOrderedMentionsBySentence.get(predictedMention.sentNum).size()==0) {
            unannotatedSpurious++;
          }
        } else {
          partitions.add(doc.goldCorefClusters.get(goldMentions.get(predictedMention.mentionID).goldCorefClusterID));
        }
      }
      pNum -= partitions.size();
      pDen += Math.max(0, c.corefMentions.size()-unannotatedSpurious-1);
    }
    // for c1+c2
    pNum += (c1.corefMentions.size()+c2.corefMentions.size());
    int unannotatedSpurious = 0;
    Set<CorefCluster> partitions = new HashSet<CorefCluster>();
    for (Mention predictedMention : c1.corefMentions){
      if(!goldMentions.containsKey(predictedMention.mentionID)) {  // twinless predictedmention
        pNum--;
        if(doc.goldOrderedMentionsBySentence.get(predictedMention.sentNum).size()==0) {
          unannotatedSpurious++;
        }
      } else {
        partitions.add(doc.goldCorefClusters.get(goldMentions.get(predictedMention.mentionID).goldCorefClusterID));
      }
    }
    for (Mention predictedMention : c2.corefMentions){
      if(!goldMentions.containsKey(predictedMention.mentionID)) {  // twinless predictedmention
        pNum--;
        if(doc.goldOrderedMentionsBySentence.get(predictedMention.sentNum).size()==0) {
          unannotatedSpurious++;
        }
      } else {
        partitions.add(doc.goldCorefClusters.get(goldMentions.get(predictedMention.mentionID).goldCorefClusterID));
      }
    }
    
    pNum -= partitions.size();
    pDen += Math.max(0, c1.corefMentions.size() + c2.corefMentions.size() -unannotatedSpurious-1);
    
    
    precisionDenSum += pDen;
    precisionNumSum += pNum;
  }
  
}
