package edu.stanford.nlp.jcoref.dcoref.sievepasses;

public class EventWNSynonymMatch extends DeterministicCorefSieve {
  public EventWNSynonymMatch() {
    super();
    flags.FOR_EVENT = true;
    flags.USE_WNSYNONYM = true;
  }
  public EventWNSynonymMatch(String args) {
    this();
    flags.set(args);
  }
}
