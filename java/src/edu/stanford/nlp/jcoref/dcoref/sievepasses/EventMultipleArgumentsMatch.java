package edu.stanford.nlp.jcoref.dcoref.sievepasses;

public class EventMultipleArgumentsMatch extends DeterministicCorefSieve {
  public EventMultipleArgumentsMatch() {
    super();
    flags.USE_iwithini = true;
    flags.EVENT_MULTIPLE_ARG_MATCH = true;
    flags.CLUSTER_MATCH = true;
    flags.FOR_EVENT = true;
  }
  public EventMultipleArgumentsMatch(String args) {
    this();
    flags.set(args);
  }
}