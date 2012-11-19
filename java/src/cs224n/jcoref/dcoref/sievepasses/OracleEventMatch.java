package edu.stanford.nlp.jcoref.dcoref.sievepasses;

public class OracleEventMatch extends DeterministicCorefSieve {
  public OracleEventMatch() {
    super();
    flags.ORACLE_EVENT = true;
    flags.FOR_EVENT = true;
  }
  public OracleEventMatch(String args) {
    this();
    flags.set(args);
  }
}
