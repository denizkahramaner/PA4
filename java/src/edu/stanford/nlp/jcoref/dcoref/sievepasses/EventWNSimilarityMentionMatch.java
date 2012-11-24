package edu.stanford.nlp.jcoref.dcoref.sievepasses;

public class EventWNSimilarityMentionMatch extends DeterministicCorefSieve {

  public EventWNSimilarityMentionMatch() {
    super();
    flags.FOR_EVENT = true;
    flags.USE_WNSIMILARITY_MENTION = true;
  }
  public EventWNSimilarityMentionMatch(String args) {
    this();
    flags.set(args);
  }
}