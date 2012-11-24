package edu.stanford.nlp.jcoref.dcoref.sievepasses;

public class EventSchemaMatch extends DeterministicCorefSieve {

  public EventSchemaMatch() {
    super();
    flags.FOR_EVENT = true;
    flags.USE_SCHEMA_MATCH = true;
  }
  public EventSchemaMatch(String args) {
    this();
    flags.set(args);
  }
}
