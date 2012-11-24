package edu.stanford.nlp.jcoref.dcoref.sievepasses;

public class EventSuperSenseMatch extends DeterministicCorefSieve {
  public EventSuperSenseMatch() {
    super();
    flags.FOR_EVENT = true;
    flags.USE_SUPERSENSE = true;
    flags.COREF_SRLARG = true;
  }
  public EventSuperSenseMatch(String args) {
    this();
    flags.set(args);
  }
}