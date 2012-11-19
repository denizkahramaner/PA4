package edu.stanford.nlp.jcoref.dcoref;

import java.util.List;

import edu.stanford.nlp.jcoref.dcoref.SieveCoreferenceSystem.Semantics;
import edu.stanford.nlp.pipeline.Annotation;

/**
 * Interface for finding coref mentions in a document
 *
 * @author Angel Chang
 */
public interface CorefMentionFinder {
  public List<List<Mention>> extractPredictedMentions(Annotation doc, int maxGoldID, Dictionaries dict, Semantics semantics);
  public List<List<Mention>> extractPredictedMentions(Annotation doc, int maxGoldID, Dictionaries dict,
      Semantics semantics, List<List<Mention>> goldMentions, boolean detectMentionsInAnnotatedSentencesOnly);
}
