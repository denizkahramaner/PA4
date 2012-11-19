package edu.stanford.nlp.jcoref;

import java.io.Serializable;
import java.util.List;

import edu.stanford.nlp.jcoref.dcoref.Dictionaries;
import edu.stanford.nlp.jcoref.dcoref.Document;
import edu.stanford.nlp.jcoref.dcoref.Mention;
import edu.stanford.nlp.jcoref.dcoref.SieveCoreferenceSystem.Semantics;
import edu.stanford.nlp.ling.CoreAnnotations.DocIDAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.Triple;

public class JointCorefDocument extends Document implements Serializable {

  private static final long serialVersionUID = -4830916817929576918L;
  public String docID;
  public List<Triple<String, Integer, Integer>> originalDocs;

  JointCorefDocument(String docID, Annotation anno,
      List<List<Mention>> predictedMentions,
      List<List<Mention>> goldMentions, Dictionaries dict, Semantics semantics) {
    super(anno, predictedMentions, goldMentions, dict, semantics);
    this.docID = docID;
  }
  JointCorefDocument(JCBDocument doc,
      List<List<Mention>> predictedMentions,
      List<List<Mention>> goldMentions, Dictionaries dict, Semantics semantics) {
    super(doc.annotation, predictedMentions, goldMentions, dict, semantics);
    this.docID = doc.docID;

    // add DocIDannotation

    int sentIdx = 0;
    int docIdx = 0;
    Triple<String, Integer, Integer> currentDoc = doc.docsInMeta.get(docIdx);
    for(CoreMap sent : this.annotation.get(SentencesAnnotation.class)){
      if(sentIdx >=currentDoc.third()) {
        currentDoc = doc.docsInMeta.get(++docIdx);
      }
      sent.set(DocIDAnnotation.class, currentDoc.first());
      List<Mention> predicted = predictedMentions.get(sentIdx);
      for(Mention p : predicted) {
        p.originalDocID = currentDoc.first();
      }
      sentIdx++;
    }
    this.originalDocs = doc.docsInMeta;
  }
  public JointCorefDocument(Document doc,
      List<List<Mention>> predictedMentions,
      List<List<Mention>> goldMentions, Dictionaries dict, Semantics semantics) {
    super(doc.annotation, predictedMentions, goldMentions, dict, semantics);
    this.docID = doc.conllDoc.getDocumentID().replaceAll("/", "-")+"-part-"+doc.conllDoc.getPartNo();
  }
}
