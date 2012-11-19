package edu.stanford.nlp.jcoref.dcoref.sievepasses;

public class EventSimpleDekangLinThesaurus extends DeterministicCorefSieve {

  public EventSimpleDekangLinThesaurus() {
    super();
    flags.FOR_EVENT = true;
    flags.USE_DEKANGLIN_SIMPLE = true;
  }
  public EventSimpleDekangLinThesaurus(String args) {
    this();
    flags.set(args);
  }
}
