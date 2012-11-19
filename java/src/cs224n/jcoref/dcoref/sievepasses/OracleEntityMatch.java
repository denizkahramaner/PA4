package edu.stanford.nlp.jcoref.dcoref.sievepasses;

public class OracleEntityMatch extends DeterministicCorefSieve {
  public OracleEntityMatch() {
    super();
    flags.ORACLE_ENTITY = true;
  }
  public OracleEntityMatch(String args) {
    this();
    flags.set(args);
  }
}
