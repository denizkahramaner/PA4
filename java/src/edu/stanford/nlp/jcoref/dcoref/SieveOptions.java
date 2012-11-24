package edu.stanford.nlp.jcoref.dcoref;

public class SieveOptions {

  //
  // options for entity coref
  //

  public boolean DO_PRONOUN;
  public boolean USE_iwithini;
  public boolean USE_APPOSITION;
  public boolean USE_PREDICATENOMINATIVES;
  public boolean USE_ACRONYM;
  public boolean USE_RELATIVEPRONOUN;
  public boolean USE_ROLEAPPOSITION;
  public boolean USE_EXACTSTRINGMATCH;
  public boolean USE_INCLUSION_HEADMATCH;
  public boolean USE_RELAXED_HEADMATCH;
  public boolean USE_INCOMPATIBLE_MODIFIER;
  public boolean USE_DEMONYM;
  public boolean USE_WORDS_INCLUSION;
  public boolean USE_ROLE_SKIP;
  public boolean USE_RELAXED_EXACTSTRINGMATCH;
  public boolean USE_ATTRIBUTES_AGREE;
  public boolean USE_WN_HYPERNYM;
  public boolean USE_WN_SYNONYM;
  public boolean USE_DIFFERENT_LOCATION;
  public boolean USE_NUMBER_IN_MENTION;
  public boolean USE_PROPERHEAD_AT_LAST;
  public boolean USE_ALIAS;
  public boolean USE_SLOT_MATCH;
  public boolean USE_DISCOURSEMATCH;
  public boolean USE_THESAURUS_ENTITY_SIMILAR;
  public double THESAURUS_ENTITY_SIMILAR_THRESHOLD;
  public boolean USE_COREF_SRLPREDICATE_MATCH;
  public boolean USE_COREF_VERB_FOR_ENTITY;
  public boolean USE_COREF_DICT;
  public boolean USE_COREF_DICT_COL1;
  public boolean USE_COREF_DICT_COL2;
  public boolean USE_COREF_DICT_COL3;
  public boolean USE_COREF_DICT_COL4;
  public boolean USE_DISTANCE;
  public boolean USE_NUMBER_ANIMACY_NE_AGREE;

  //
  // options for event coref
  //

  public boolean FOR_EVENT;   // true for all event sieves
  public boolean CLUSTER_MATCH;   // true if sieve do cluster match (not mention)

  // positive constraints

  public boolean USE_LEMMAMATCH;
  public boolean USE_WNSIMILARITY_CLUSTER;
  public boolean USE_WNSIMILARITY_MENTION;
  public boolean USE_WNSYNONYM;
  public boolean USE_WNSYNSET_CLUSTER;
  public boolean USE_SCHEMA_MATCH;
  public boolean ORACLE_ENTITY;
  public boolean ORACLE_EVENT;
  public boolean USE_DEKANGLIN_SIMPLE;
  public boolean USE_SUPERSENSE;
  public boolean DO_HEADSHARING;
  public boolean USE_WNSIMILARITY_SURFACE_CONTEXT;
  public boolean USE_SRLARGUMENT_MATCH;

  // negative constraints
  public boolean WITHIN_DOC;
  public boolean USE_EVENT_iwithini;
  public boolean USE_ARG_SHARE;
  public boolean USE_COREF_ARG;
  public boolean USE_COREF_IN_SENTENCE;
  public boolean COREF_PRED;
  public boolean USE_NOT_COREF_ARG;
  public boolean USE_NOT_COREF_SRLARG;
  public boolean USE_LEMMA_OBJ;
  public boolean USE_POSSIBLE_COREF_SRLARG;
  public boolean USE_SENT_SIMILAR;
  public boolean USE_THESAURUS_SIMILAR_SRLARG;
  public double THESAURUS_SIMILAR_THRESHOLD;   // threshold for USE_THESAURUS_SIMILAR_SRLARG
  public boolean USE_NOT_FREQ_VERB;
  public double FREQ_IDF_CUTOFF;
  public double WNSIMILARITY_CLUSTER_THRESHOLD;
  public boolean USE_COREF_LEFTMENTION;
  public boolean USE_COREF_RIGHTMENTION;
  public boolean USE_NOUNEVENTS_MATCH_ONLY;
  public boolean MATCH_NOSRLARGEVENT_ONLY;
  public boolean USE_NO_COMMON_SRLARG_MATCH;
  public double CLUSTER_SYNONYM_THRESHOLD;
  public double CLUSTER_SIMILAR_THRESHOLD;
  public boolean COREF_SRLARG;
  public boolean SIMILAR_SRLARG;
  public boolean SAMEHEAD_SRLARG;
  public boolean SIMILAR_SRLPRED;
  public boolean COREF_SRLPRED;
  public boolean EVENT_LOCATION;
  public boolean EVENT_TIME;
  public boolean EVENT_MULTIPLE_ARG_MATCH;
  public double EVENT_ARG_MATCH_THRES;
  public boolean ENTITY_MULTIPLE_ARG_MATCH;
  public double ENTITY_ARG_MATCH_THRES;
  public boolean JOINT_ARG_MATCH;
  public boolean ALLOW_VERB_NOUN_MATCH;
  public boolean EVENT_ARGUMENT_MATCH;
  public boolean ENTITY_PREDICATE_MATCH;

  @Override
  public String toString() {
    StringBuilder os = new StringBuilder();
    os.append("{");
    if(WITHIN_DOC) os.append("WITHIN_DOC");

    if(DO_PRONOUN) os.append(", DO_PRONOUN");
    if(USE_iwithini) os.append(", USE_iwithini");
    if(USE_APPOSITION) os.append(", USE_APPOSITION");
    if(USE_PREDICATENOMINATIVES) os.append(", USE_PREDICATENOMINATIVES");
    if(USE_ACRONYM) os.append(", USE_ACRONYM");
    if(USE_RELATIVEPRONOUN) os.append(", USE_RELATIVEPRONOUN");
    if(USE_ROLEAPPOSITION) os.append(", USE_ROLEAPPOSITION");
    if(USE_EXACTSTRINGMATCH) os.append(", USE_EXACTSTRINGMATCH");
    if(USE_INCLUSION_HEADMATCH) os.append(", USE_INCLUSION_HEADMATCH");
    if(USE_RELAXED_HEADMATCH) os.append(", USE_RELAXED_HEADMATCH");
    if(USE_INCOMPATIBLE_MODIFIER) os.append(", USE_INCOMPATIBLE_MODIFIER");
    if(USE_DEMONYM) os.append(", USE_DEMONYM");
    if(USE_WORDS_INCLUSION) os.append(", USE_WORDS_INCLUSION");
    if(USE_ROLE_SKIP) os.append(", USE_ROLE_SKIP");
    if(USE_RELAXED_EXACTSTRINGMATCH) os.append(", USE_RELAXED_EXACTSTRINGMATCH");
    if(USE_ATTRIBUTES_AGREE) os.append(", USE_ATTRIBUTES_AGREE");
    if(USE_WN_HYPERNYM) os.append(", USE_WN_HYPERNYM");
    if(USE_WN_SYNONYM) os.append(", USE_WN_SYNONYM");
    if(USE_DIFFERENT_LOCATION) os.append(", USE_DIFFERENT_LOCATION");
    if(USE_NUMBER_IN_MENTION) os.append(", USE_NUMBER_IN_MENTION");
    if(USE_PROPERHEAD_AT_LAST) os.append(", USE_PROPERHEAD_AT_LAST");
    if(USE_ALIAS) os.append(", USE_ALIAS");
    if(USE_SLOT_MATCH) os.append(", USE_SLOT_MATCH");
    if(USE_DISCOURSEMATCH) os.append(", USE_DISCOURSEMATCH");
    if(USE_COREF_DICT) os.append(", USE_COREF_DICT");
    if(USE_COREF_DICT_COL1) os.append(", USE_COREF_DICT_COL1");
    if(USE_COREF_DICT_COL2) os.append(", USE_COREF_DICT_COL2");
    if(USE_COREF_DICT_COL3) os.append(", USE_COREF_DICT_COL3");
    if(USE_COREF_DICT_COL4) os.append(", USE_COREF_DICT_COL4");
    if(USE_DISTANCE) os.append(", USE_DISTANCE");
    if(USE_NUMBER_ANIMACY_NE_AGREE) os.append(", USE_NUMBER_ANIMACY_NE_AGREE");

    if(FOR_EVENT) os.append(", FOR_EVENT");
    if(USE_LEMMAMATCH) os.append(", USE_LEMMAMATCH");
    if(USE_ARG_SHARE) os.append(", USE_ARG_SHARE");
    if(USE_COREF_ARG) os.append(", USE_COREF_ARG");
    if(USE_WNSIMILARITY_CLUSTER) os.append(", USE_WNSIMILARITY_CLUSTER");
    if(USE_WNSIMILARITY_MENTION) os.append(", USE_WNSIMILARITY_MENTION");
    if(USE_WNSYNONYM) os.append(", USE_WNSYNONYM");
    if(USE_WNSYNSET_CLUSTER) os.append(", USE_WNSYNSET_CLUSTER");
    if(USE_SCHEMA_MATCH) os.append(", USE_SCHEMA_MATCH");
    if(ORACLE_ENTITY) os.append(", ORACLE_ENTITY");
    if(ORACLE_EVENT) os.append(", ORACLE_EVENT");
    if(USE_COREF_IN_SENTENCE) os.append(", USE_COREF_IN_SENTENCE");
    if(USE_NOT_COREF_ARG) os.append(", USE_NOT_COREF_ARG");
    if(USE_NOT_COREF_SRLARG) os.append(", USE_NOT_COREF_SRLARG");
    if(USE_LEMMA_OBJ) os.append(", USE_LEMMA_OBJ");
    if(COREF_SRLARG) os.append(", USE_COREF_SRLARG");
    if(DO_HEADSHARING) os.append(", DO_HEADSHARING");

    if(COREF_PRED) os.append(", COREF_PRED");
    if(USE_COREF_SRLPREDICATE_MATCH) os.append(", USE_COREF_SRLPREDICATE_MATCH");
    if(USE_POSSIBLE_COREF_SRLARG) os.append(", USE_POSSIBLE_COREF_SRLARG");
    if(USE_DEKANGLIN_SIMPLE) os.append(", USE_DEKANGLIN_SIMPLE");
    if(USE_SUPERSENSE) os.append(", USE_SUPERSENSE");
    if(USE_SENT_SIMILAR) os.append(", USE_SENT_SIMILAR");
    if(USE_EVENT_iwithini) os.append(", USE_EVENT_iwithini");
    if(USE_THESAURUS_SIMILAR_SRLARG) os.append(", USE_THESAURUS_SIMILAR_SRLARG");
    if(USE_THESAURUS_SIMILAR_SRLARG) os.append(", THESAURUS_SIMILAR_THRESHOLD="+THESAURUS_SIMILAR_THRESHOLD);
    if(USE_THESAURUS_ENTITY_SIMILAR) os.append(", USE_THESAURUS_ENTITY_SIMILAR");
    if(THESAURUS_ENTITY_SIMILAR_THRESHOLD!=0) os.append(", THESAURUS_ENTITY_SIMILAR_THRESHOLD="+THESAURUS_ENTITY_SIMILAR_THRESHOLD);
    if(USE_NOT_FREQ_VERB) os.append(", USE_NOT_FREQ_VERB");
    if(USE_NOT_FREQ_VERB) os.append(", FREQ_IDF_CUTOFF="+FREQ_IDF_CUTOFF);
    if(USE_WNSIMILARITY_CLUSTER) os.append(", WNSIMILARITY_CLUSTER_THRESHOLD="+WNSIMILARITY_CLUSTER_THRESHOLD);
    if(USE_COREF_LEFTMENTION) os.append(", USE_COREF_LEFTMENTION");
    if(USE_COREF_RIGHTMENTION) os.append(", USE_COREF_RIGHTMENTION");
    if(USE_WNSIMILARITY_SURFACE_CONTEXT) os.append(", USE_WNSIMILARITY_SURFACE_CONTEXT");
    if(USE_SRLARGUMENT_MATCH) os.append(", USE_SRLARGUMENT_MATCH");
    if(USE_COREF_VERB_FOR_ENTITY) os.append(", USE_COREF_VERB_FOR_ENTITY");
    if(USE_NOUNEVENTS_MATCH_ONLY) os.append(", USE_NOUNEVENTS_MATCH_ONLY");
    if(CLUSTER_MATCH) os.append(", CLUSTER_MATCH");
    if(MATCH_NOSRLARGEVENT_ONLY) os.append(", MATCH_NOSRLARGEVENT_ONLY");
    if(USE_NO_COMMON_SRLARG_MATCH) os.append(", USE_NO_COMMON_SRLARG_MATCH");
    if(CLUSTER_SYNONYM_THRESHOLD!=-1) os.append(", CLUSTER_SYNONYM_THRESHOLD="+CLUSTER_SYNONYM_THRESHOLD);
    if(CLUSTER_SIMILAR_THRESHOLD!=-1) os.append(", CLUSTER_SIMILAR_THRESHOLD="+CLUSTER_SIMILAR_THRESHOLD);
    if(SIMILAR_SRLARG) os.append(", SIMILAR_SRLARG");
    if(SAMEHEAD_SRLARG) os.append(", SAMEHEAD_SRLARG");
    if(COREF_SRLPRED) os.append(", COREF_SRLPRED");
    if(SIMILAR_SRLPRED) os.append(", SIMILAR_SRLPRED");
    if(EVENT_LOCATION) os.append(", EVENT_LOCATION");
    if(EVENT_TIME) os.append(", EVENT_TIME");
    if(EVENT_MULTIPLE_ARG_MATCH) os.append(", EVENT_MULTIPLE_ARG_MATCH");
    if(EVENT_ARG_MATCH_THRES!=-1) os.append(", EVENT_ARG_MATCH_THRES="+EVENT_ARG_MATCH_THRES);
    if(ENTITY_MULTIPLE_ARG_MATCH) os.append(", ENTITY_MULTIPLE_ARG_MATCH");
    if(ENTITY_ARG_MATCH_THRES!=-1) os.append(", ENTITY_ARG_MATCH_THRES="+ENTITY_ARG_MATCH_THRES);
    if(JOINT_ARG_MATCH) os.append(", JOINT_ARG_MATCH");
    if(ALLOW_VERB_NOUN_MATCH) os.append(", ALLOW_VERB_NOUN_MATCH");
    
    if(EVENT_ARGUMENT_MATCH) os.append(", EVENT_ARGUMENT_MATCH");
    if(ENTITY_PREDICATE_MATCH) os.append(", ENTITY_PREDICATE_MATCH");

    os.append("}");
    return os.toString();
  }

  public SieveOptions() {
    WITHIN_DOC = false;
    DO_PRONOUN = false;
    USE_iwithini = false;
    USE_APPOSITION = false;
    USE_PREDICATENOMINATIVES = false;
    USE_ACRONYM = false;
    USE_RELATIVEPRONOUN = false;
    USE_ROLEAPPOSITION = false;
    USE_EXACTSTRINGMATCH = false;
    USE_INCLUSION_HEADMATCH = false;
    USE_RELAXED_HEADMATCH = false;
    USE_INCOMPATIBLE_MODIFIER = false;
    USE_DEMONYM = false;
    USE_WORDS_INCLUSION = false;
    USE_ROLE_SKIP = false;
    USE_RELAXED_EXACTSTRINGMATCH = false;
    USE_ATTRIBUTES_AGREE = false;
    USE_WN_HYPERNYM = false;
    USE_WN_SYNONYM = false;
    USE_DIFFERENT_LOCATION = false;
    USE_NUMBER_IN_MENTION = false;
    USE_PROPERHEAD_AT_LAST = false;
    USE_ALIAS = false;
    USE_SLOT_MATCH = false;
    USE_DISCOURSEMATCH = false;
    USE_COREF_DICT = false;
    USE_COREF_DICT_COL1 = false;
    USE_COREF_DICT_COL2 = false;
    USE_COREF_DICT_COL3 = false;
    USE_COREF_DICT_COL4 = false;
    USE_DISTANCE = false;
    USE_NUMBER_ANIMACY_NE_AGREE = false;

    FOR_EVENT = false;
    USE_LEMMAMATCH = false;
    USE_ARG_SHARE = false;
    USE_COREF_ARG = false;
    USE_WNSIMILARITY_CLUSTER = false;
    USE_WNSIMILARITY_MENTION = false;
    USE_WNSYNONYM = false;
    USE_WNSYNSET_CLUSTER = false;
    USE_SCHEMA_MATCH = false;
    ORACLE_ENTITY = false;
    ORACLE_EVENT = false;
    USE_COREF_IN_SENTENCE = false;
    USE_NOT_COREF_ARG = false;
    USE_LEMMA_OBJ = false;
    DO_HEADSHARING = false;
    USE_COREF_SRLPREDICATE_MATCH = false;
    USE_POSSIBLE_COREF_SRLARG = false;
    USE_DEKANGLIN_SIMPLE = false;
    USE_SUPERSENSE = false;
    USE_SENT_SIMILAR = false;
    USE_EVENT_iwithini = false;
    USE_THESAURUS_SIMILAR_SRLARG = false;
    THESAURUS_SIMILAR_THRESHOLD = 0;
    USE_NOT_COREF_SRLARG = false;
    USE_THESAURUS_ENTITY_SIMILAR = false;
    THESAURUS_ENTITY_SIMILAR_THRESHOLD = 0;
    USE_NOT_FREQ_VERB = false;
    FREQ_IDF_CUTOFF = 0;
    WNSIMILARITY_CLUSTER_THRESHOLD = 1;
    USE_COREF_LEFTMENTION = false;
    USE_COREF_RIGHTMENTION = false;
    USE_WNSIMILARITY_SURFACE_CONTEXT = false;
    USE_SRLARGUMENT_MATCH = false;
    USE_COREF_VERB_FOR_ENTITY = false;
    USE_NOUNEVENTS_MATCH_ONLY = false;
    CLUSTER_MATCH = false;
    MATCH_NOSRLARGEVENT_ONLY = false;
    USE_NO_COMMON_SRLARG_MATCH = false;
    CLUSTER_SYNONYM_THRESHOLD = -1;
    CLUSTER_SIMILAR_THRESHOLD = -1;
    COREF_PRED = false;

    COREF_SRLARG = false;
    SIMILAR_SRLARG = false;
    SAMEHEAD_SRLARG = false;
    SIMILAR_SRLPRED = false;
    COREF_SRLPRED = false;

    EVENT_LOCATION = false;
    EVENT_TIME = false;
    EVENT_MULTIPLE_ARG_MATCH = false;
    EVENT_ARG_MATCH_THRES = -1;
    ENTITY_MULTIPLE_ARG_MATCH = false;
    ENTITY_ARG_MATCH_THRES = -1;
    JOINT_ARG_MATCH = false;
    ALLOW_VERB_NOUN_MATCH = false;
    
    EVENT_ARGUMENT_MATCH = false;
    ENTITY_PREDICATE_MATCH = false;
  }

  public void set(String args) {
    for(String arg : args.split(",")) {
      if(arg.equals("ENTITY_iwithini")) {
        this.USE_iwithini = true;
      } else if(arg.equals("COREF_SRLARG")) {
        this.COREF_SRLARG = true;
      } else if (arg.equals("POSSIBLE_COREF_SRLARG")) {
        this.USE_POSSIBLE_COREF_SRLARG = true;
      } else if (arg.equals("COREF_ARG")) {
        this.USE_COREF_ARG = true;
      } else if (arg.equals("COREF_IN_SENTENCE")) {
        this.USE_COREF_IN_SENTENCE = true;
      } else if (arg.equals("SENT_SIMILAR")) {
        this.USE_SENT_SIMILAR = true;
      } else if (arg.equals("NOT_COREF_ARG")) {
        this.USE_NOT_COREF_ARG = true;
      } else if (arg.equals("NOT_COREF_SRLARG")) {
        this.USE_NOT_COREF_SRLARG = true;
      } else if (arg.equals("EVENT_iwithini")) {
        this.USE_EVENT_iwithini = true;
      } else if (arg.equals("USE_COREF_LEFTMENTION")) {
        this.USE_COREF_LEFTMENTION = true;
      } else if (arg.equals("USE_COREF_RIGHTMENTION")) {
        this.USE_COREF_RIGHTMENTION = true;
      } else if (arg.equals("USE_WNSIMILARITY_SURFACE_CONTEXT")) {
        this.USE_WNSIMILARITY_SURFACE_CONTEXT = true;
      } else if (arg.startsWith("THESAURUS_SIMILAR_SRLARG")) {
        double threshold = Double.parseDouble(arg.split(":")[1]);
        this.USE_THESAURUS_SIMILAR_SRLARG = true;
        this.THESAURUS_SIMILAR_THRESHOLD = threshold;
      } else if (arg.startsWith("THESAURUS_ENTITY_SIMILAR")) {
        double threshold = Double.parseDouble(arg.split(":")[1]);
        this.THESAURUS_ENTITY_SIMILAR_THRESHOLD = threshold;
      } else if (arg.startsWith("NOT_FREQ_VERB")) {
        double cutoff = Double.parseDouble(arg.split(":")[1]);
        this.USE_NOT_FREQ_VERB = true;
        this.FREQ_IDF_CUTOFF = cutoff;
      } else if (arg.startsWith("WNSIMILARITY_CLUSTER_THRESHOLD")) {
        double cutoff = Double.parseDouble(arg.split(":")[1]);
        this.WNSIMILARITY_CLUSTER_THRESHOLD = cutoff;
      } else if (arg.equals("USE_COREF_VERB_FOR_ENTITY")) {
        this.USE_COREF_VERB_FOR_ENTITY = true;
      } else if (arg.equals("NOUNEVENTS_MATCH_ONLY")) {
        this.USE_NOUNEVENTS_MATCH_ONLY = true;
      } else if (arg.equals("MATCH_NOSRLARGEVENT_ONLY")) {
        this.MATCH_NOSRLARGEVENT_ONLY = true;
      } else if (arg.equals("USE_NO_COMMON_SRLARG_MATCH")) {
        this.USE_NO_COMMON_SRLARG_MATCH = true;
      } else if (arg.startsWith("CLUSTER_SYNONYM_THRESHOLD")) {
        double cutoff = Double.parseDouble(arg.split(":")[1]);
        this.CLUSTER_SYNONYM_THRESHOLD = cutoff;
      } else if (arg.startsWith("CLUSTER_SIMILAR_THRESHOLD")) {
        double cutoff = Double.parseDouble(arg.split(":")[1]);
        this.CLUSTER_SIMILAR_THRESHOLD = cutoff;
      } else if (arg.equals("WITHIN_DOC")) {
        this.WITHIN_DOC = true;
      } else if (arg.equals("COREF_SRLPRED")) {
        this.COREF_SRLPRED = true;
      } else if (arg.equals("SIMILAR_SRLPRED")) {
        this.SIMILAR_SRLPRED = true;
      } else if (arg.equals("SIMILAR_SRLARG")) {
        this.SIMILAR_SRLARG = true;
      } else if (arg.equals("SAMEHEAD_SRLARG")) {
        this.SAMEHEAD_SRLARG = true;
      } else if (arg.equals("EVENT_LOCATION")) {
        this.EVENT_LOCATION = true;
      } else if (arg.equals("EVENT_TIME")) {
        this.EVENT_TIME = true;
      } else if (arg.startsWith("EVENT_ARG_MATCH_THRES")) {
        double thres = Double.parseDouble(arg.split(":")[1]);
        this.EVENT_ARG_MATCH_THRES = thres;
      } else if (arg.startsWith("ENTITY_ARG_MATCH_THRES")) {
        double thres = Double.parseDouble(arg.split(":")[1]);
        this.ENTITY_ARG_MATCH_THRES = thres;
      } else if (arg.equals("ALLOW_VERB_NOUN_MATCH")) {
        this.ALLOW_VERB_NOUN_MATCH = true;
      } else if (arg.equals("USE_COREF_DICT")) {
        this.USE_COREF_DICT = true;
      } else if (arg.equals("USE_COREF_DICT_COL1")) {
        this.USE_COREF_DICT_COL1 = true;
      } else if (arg.equals("USE_COREF_DICT_COL2")) {
        this.USE_COREF_DICT_COL2 = true;
      } else if (arg.equals("USE_COREF_DICT_COL3")) {
        this.USE_COREF_DICT_COL3 = true;
      } else if (arg.equals("USE_COREF_DICT_COL4")) {
        this.USE_COREF_DICT_COL4 = true;
      } else if (arg.equals("USE_DISTANCE")) {
        this.USE_DISTANCE = true;
      } else if (arg.equals("USE_NUMBER_ANIMACY_NE_AGREE")) {
        this.USE_NUMBER_ANIMACY_NE_AGREE = true;
      } else if (arg.startsWith("CLASSIFIER_CUTOFF")) {
        // do nothing : this is used in JointArgumentMatch constructor
      } else {
        throw new RuntimeException("SieveOptions.java: incorrect sieve option: "+arg);
      }
    }
  }
}
