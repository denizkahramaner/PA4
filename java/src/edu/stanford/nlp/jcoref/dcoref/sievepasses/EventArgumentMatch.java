package edu.stanford.nlp.jcoref.dcoref.sievepasses;

public class EventArgumentMatch extends DeterministicCorefSieve {
  public EventArgumentMatch() {
    super();
    flags.FOR_EVENT = true;
    flags.EVENT_ARGUMENT_MATCH = true;
    flags.CLUSTER_MATCH = true;
  }
  public EventArgumentMatch(String args) {
    this();
    flags.set(args);
  }
}