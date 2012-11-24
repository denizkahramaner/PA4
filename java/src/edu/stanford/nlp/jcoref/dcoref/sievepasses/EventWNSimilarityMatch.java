package edu.stanford.nlp.jcoref.dcoref.sievepasses;

public class EventWNSimilarityMatch extends DeterministicCorefSieve {

  public EventWNSimilarityMatch() {
    super();
    flags.FOR_EVENT = true;
    flags.CLUSTER_MATCH = true;
    flags.USE_WNSIMILARITY_CLUSTER = true;
  }
  public EventWNSimilarityMatch(String args) {
    this();
    flags.set(args);
  }
}
