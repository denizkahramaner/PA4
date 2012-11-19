package edu.stanford.nlp.jcoref.dcoref.sievepasses;

public class EventWNSimilaritySurfaceContextMatch extends DeterministicCorefSieve {

  public EventWNSimilaritySurfaceContextMatch() {
    super();
    flags.FOR_EVENT = true;
    flags.USE_WNSIMILARITY_SURFACE_CONTEXT = true;
  }
  public EventWNSimilaritySurfaceContextMatch(String args) {
    this();
    flags.set(args);
  }
}