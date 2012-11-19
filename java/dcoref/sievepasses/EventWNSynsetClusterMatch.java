package edu.stanford.nlp.jcoref.dcoref.sievepasses;

public class EventWNSynsetClusterMatch extends DeterministicCorefSieve {
  public EventWNSynsetClusterMatch() {
    super();
    flags.FOR_EVENT = true;
    flags.USE_WNSYNSET_CLUSTER = true;
  }
  public EventWNSynsetClusterMatch(String args) {
    this();
    flags.set(args);
  }
}
