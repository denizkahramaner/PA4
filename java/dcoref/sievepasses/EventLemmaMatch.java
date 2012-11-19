package edu.stanford.nlp.jcoref.dcoref.sievepasses;

public class EventLemmaMatch extends DeterministicCorefSieve {
  public EventLemmaMatch() {
    super();
    flags.FOR_EVENT = true;
    flags.USE_LEMMAMATCH = true;
    flags.CLUSTER_MATCH = true;
  }
  public EventLemmaMatch(String args) {
    this();
    flags.set(args);
  }
}