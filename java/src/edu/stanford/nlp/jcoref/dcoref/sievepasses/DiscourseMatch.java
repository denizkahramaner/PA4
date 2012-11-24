package edu.stanford.nlp.jcoref.dcoref.sievepasses;

public class DiscourseMatch extends DeterministicCorefSieve {
  public DiscourseMatch() {
    super();
    flags.USE_DISCOURSEMATCH = true;
  }
  public DiscourseMatch(String args) {
    this();
    flags.set(args);
  }
}
