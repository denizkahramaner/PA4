package edu.stanford.nlp.jcoref.dcoref.sievepasses;

public class MarkRole extends DeterministicCorefSieve {
  public MarkRole() {
    super();
    flags.USE_ROLE_SKIP = true;
  }
}
