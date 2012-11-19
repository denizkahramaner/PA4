package edu.stanford.nlp.jcoref;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import edu.stanford.nlp.jcoref.dcoref.Mention;
import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TextAnnotation;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.IntPair;
import edu.stanford.nlp.util.Triple;

/** output of JCBReader */
public class JCBDocument implements Serializable {
  private static final long serialVersionUID = 7461693458768055019L;
  public String docID;
  public Annotation annotation;
  public List<List<Mention>> goldMentions;
  public int maxIDGoldMention;
  public List<Map<Integer, Map<IntPair, String>>> srlInfo;
  public List<Triple<String, Integer, Integer>> docsInMeta;

  JCBDocument(String docID) {
    this.docID = docID;
    this.maxIDGoldMention = -1;
    annotation = new Annotation("");
    annotation.set(SentencesAnnotation.class, new ArrayList<CoreMap>());
    goldMentions = new ArrayList<List<Mention>>();
  }

  /** constructor for building meta doc */
  JCBDocument(String docID, List<JCBDocument> docs){
    this(docID);

    List<CoreMap> sentences = new ArrayList<CoreMap>();
    StringBuilder text = new StringBuilder();
    srlInfo = new ArrayList<Map<Integer, Map<IntPair, String>>>();
    docsInMeta = new ArrayList<Triple<String, Integer, Integer>>();

    for(JCBDocument doc : docs){
      goldMentions.addAll(doc.goldMentions);
      text.append(doc.annotation.get(TextAnnotation.class)).append("\n");
      sentences.addAll(doc.annotation.get(SentencesAnnotation.class));
      if(maxIDGoldMention < doc.maxIDGoldMention) maxIDGoldMention = doc.maxIDGoldMention;
      srlInfo.addAll(doc.srlInfo);
      docsInMeta.add(new Triple<String, Integer, Integer>(doc.docID, sentences.size()-doc.annotation.get(SentencesAnnotation.class).size(), sentences.size()));
    }
    annotation.set(TextAnnotation.class, text.toString());
    annotation.set(SentencesAnnotation.class, sentences);
  }

  @Override
  public String toString() {
    return docID;
  }
}
