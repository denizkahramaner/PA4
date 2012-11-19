package edu.stanford.nlp.jcoref.dcoref;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.*;

import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.IndexedWord;
import edu.stanford.nlp.ling.CoreAnnotations.*;
import edu.stanford.nlp.trees.GrammaticalRelation;
import edu.stanford.nlp.trees.EnglishGrammaticalRelations;
import edu.stanford.nlp.trees.semgraph.SemanticGraph;
import edu.stanford.nlp.util.CoreMap;

import edu.stanford.nlp.jcoref.dcoref.Dictionaries.Person;
import edu.stanford.nlp.util.Pair;

/**
 *
 * @author Marie-Catherine de Marneffe
 * @author Marta Recasens
 */
public class EntityLifespan {

  public static final Set<String> determiners = new HashSet<String>(Arrays.asList("the", "this", "that", "these", "those", "his", "her", "my", "your", "their", "our"));

  public static final Set<String> negations = new HashSet<String>(Arrays.asList("n't","not", "nor", "neither", "never", "no", "non", "any", "none", "nobody", "nothing", "nowhere", "nearly","almost",
      "if",
      "false", "fallacy", "unsuccessfully", "unlikely", "impossible", "improbable", "uncertain", "unsure", "impossibility", "improbability",
      "cancellation", "breakup", "lack", "long-stalled", "end", "rejection", "failure", "avoid", "bar", "block", "break", "cancel", "cease", "cut", "decline", "deny", "deprive", "destroy", "excuse",
      "fail", "forbid", "forestall", "forget", "halt", "lose", "nullify", "prevent", "refrain", "reject",
      "rebut", "remain", "refuse", "stop", "suspend", "ward"));

  public static final Set<String> neg_relations = new HashSet<String>(Arrays.asList("prep_without", "prepc_without", "prep_except", "prepc_except", "prep_excluding", "prepx_excluding",
      "prep_if", "prepc_if", "prep_whether", "prepc_whether", "prep_away_from", "prepc_away_from", "prep_instead_of", "prepc_instead_of"));

  public static final Set<String> modals = new HashSet<String>(Arrays.asList("can", "could", "may", "might", "must", "should", "would", "seem",
      "able", "apparently", "necessarily", "presumably", "probably", "possibly", "reportedly", "supposedly",
      "inconceivable", "chance", "impossibility", "improbability", "encouragement",
      "improbable", "impossible", "likely", "necessary", "probable", "possible", "uncertain", "unlikely", "unsure",
      "likelihood", "probability", "possibility",
      "eventual", "hypothetical" , "presumed", "supposed", "reported", "apparent"));

  public static final Set<String> quantifiers = new HashSet<String>(Arrays.asList("all", "both", "neither", "either"));

  public static final Set<String> reportVerb = new HashSet<String>(Arrays.asList(
      "accuse", "acknowledge", "add", "admit", "advise", "agree", "alert",
      "allege", "announce", "answer", "apologize", "argue",
      "ask", "assert", "assure", "beg", "believe", "blame", "boast",
      "certify", "charge", "cite", "claim", "clarify", "command", "comment",
      "compare", "complain", "concede", "conclude", "confirm", "confront", "congratulate",
      "contend", "contradict", "convey", "convict", "counter", "criticize",
      "debate", "decide", "declare", "deduce", "defend", "deliver", "demand", "demonstrate", "deny",
      "describe", "determine", "disagree", "disclose", "discount", "discover", "discuss",
      "dismiss", "dispute", "disregard", "divulge", "doubt", "emphasize", "encourage", "endorse", "ensure",
      "estimate", "expect", "explain", "express", "extoll", "fear", "feel",
      "find", "forbid", "forecast", "foretell", "forget", "gather", "guarantee", "guess",
      "hear", "hint", "hope", "illustrate", "imagine", "imply", "indicate", "inform",
      "insert", "insist", "instruct", "interpret", "interview", "invite",
      "justify", "maintain", "mean", "mention", "negotiate", "note", "notify",
      "observe", "offer", "order", "persuade", "pledge", "point out",
      "praise", "pray", "predict", "prefer", "present", "promise", "propose",
      "protest", "prove", "provoke", "publish", "question", "quote", "raise", "rally", "read",
      "reaffirm", "realise", "realize", "reassure", "rebut", "recall", "reckon", "recommend",
      "reflect", "refuse", "refute", "reiterate", "reject", "relate", "remark",
      "remember", "remind", "repeat", "reply", "report", "request", "respond",
      "restate", "reveal", "say", "show", "signal",
      "slam", "speculate", "spoke", "spread", "state", "stipulate", "stress",
      "suggest", "support", "suppose", "surmise", "suspect", "swear", 
      "tell", "testify", "think", "threaten", "told", "uncover", "underline",
      "underscore", "urge", "vow", "warn", "wish", "wonder", "worry", "write",
      "applaud", "assail", "attack", "attribute", "condemn", "decry", 
      "denounce", "deplore", "enjoy", "exclude", "excuse", "experience", "forgive", "gloss",
      "hail", "hate", "investigate", "lament", "laugh", "look", "love", "notice",
      "presume", "protect", "recognize", "regret", "resent", "rumor", "sketch", "specify", "spell",
      "surprise", "value", "view", "watch", "witness"));  

  public static final Set<String> reportNoun = new HashSet<String>(Arrays.asList(
      "ABCs", "acclamation", "account", "accusation", "acknowledgment", "address", "addressing",
      "admission", "advertisement", "advice", "advisory", "affidavit", "affirmation", "alert",
      "allegation", "analysis", "anecdote", "annotation", "announcement", "answer", "antiphon",
      "apology", "applause", "appreciation", "argument", "arraignment", "article", "articulation",
      "aside", "assertion", "asseveration", "assurance", "attestation", "attitude",
      "averment", "avouchment", "avowal", "axiom", "backcap", "band-aid", "basic", "belief", "bestowal",
      "bill", "blame", "blow-by-blow", "bomb", "book", "bow", "break", "breakdown", "brief", "briefing",
      "broadcast", "broadcasting", "bulletin", "buzz", "cable", "calendar", "call", "canard", "canon",
      "card", "cause", "censure", "certification", "characterization", "charge", "chat", "chatter",
      "chitchat", "chronicle", "chronology", "citation", "claim", "clarification", "close", "cognizance",
      "comeback", "comment", "commentary", "communication", "communique", "composition", "concept",
      "concession", "conference", "confession", "confirmation", "conjecture", "connotation", "construal",
      "construction", "consultation", "contention", "contract", "convention", "conversation", "converse",
      "conviction", "counterclaim", "credenda", "creed", "critique",
      "cry", "declaration", "defense", "definition", "delineation", "delivery", "demonstration",
      "denial", "denotation", "depiction", "deposition", "description", "detail", "details", "detention",
      "dialogue", "diction", "dictum", "digest", "directive", "disclosure", "discourse", "discovery",
      "discussion", "dispatch", "display", "disquisition", "dissemination", "dissertation", "divulgence",
      "dogma", "editorial", "ejaculation", "emphasis", "enlightenment",
      "enunciation", "essay", "evidence", "examination", "example", "excerpt", "exclamation",
      "excuse", "execution", "exegesis", "explanation", "explication", "exposing", "exposition", "expounding",
      "expression", "eye-opener", "feedback", "fiction", "findings", "fingerprint", "flash", "formulation",
      "fundamental", "gift", "gloss", "goods", "gospel", "gossip", "gratitude", "greeting",
      "guarantee", "hail", "hailing", "handout", "hash", "headlines", "hearing", "hearsay",
      "ideas", "idiom", "illustration", "impeachment", "implantation", "implication", "imputation",
      "incrimination", "indication", "indoctrination", "inference", "info", "information",
      "innuendo", "insinuation", "insistence", "instruction", "intelligence", "interpretation", "interview",
      "intimation", "intonation", "issue", "item", "itemization", "justification", "key", "knowledge",
      "leak", "letter", "locution", "manifesto",
      "meaning", "meeting", "mention", "message", "missive", "mitigation", "monograph", "motive", "murmur",
      "narration", "narrative", "news", "nod", "note", "notice", "notification", "oath", "observation",
      "okay", "opinion", "oral", "outline", "paper", "parley", "particularization", "phrase", "phraseology",
      "phrasing", "picture", "piece", "pipeline", "pitch", "plea", "plot", "portraiture", "portrayal",
      "position", "potboiler", "prating", "precept", "prediction", "presentation", "presentment", "principle",
      "proclamation", "profession", "program", "promulgation", "pronouncement", "pronunciation", "propaganda",
      "prophecy", "proposal", "proposition", "prosecution", "protestation", "publication", "publicity",
      "publishing", "quotation", "ratification", "reaction", "reason", "rebuttal", "receipt", "recital",
      "recitation", "recognition", "record", "recount", "recountal", "refutation", "regulation", "rehearsal",
      "rejoinder", "relation", "release", "remark", "rendition", "repartee", "reply", "report", "reporting",
      "representation", "resolution", "response", "result", "retort", "return", "revelation", "review",
      "rule", "rumble", "rumor", "rundown", "saying", "scandal", "scoop",
      "scuttlebutt", "sense", "showing", "sign", "signature", "significance", "sketch", "skinny", "solution",
      "speaking", "specification", "speech", "statement", "story", "study", "style", "suggestion",
      "summarization", "summary", "summons", "tale", "talk", "talking", "tattle", "telecast",
      "telegram", "telling", "tenet", "term", "testimonial", "testimony", "text", "theme", "thesis",
      "tract", "tractate", "tradition", "translation", "treatise", "utterance", "vent", "ventilation",
      "verbalization", "version", "vignette", "vindication", "warning",
      "warrant", "whispering", "wire", "word", "work", "writ", "write-up", "writeup", "writing",
      "acceptance", "complaint", "concern", "disappointment", "disclose", "estimate", "laugh", "pleasure", "regret",
      "resentment", "view"));

  private static void getFeatures(BufferedWriter out, Document doc, Mention men, IndexedWord head) throws IOException{    

    /*System.out.println("MEN: " + men);
    System.out.println(Mention.sentenceWordsToString(men));
    System.out.println("HEAD: " + men.headString);
    System.out.println(men.dependency);
    System.out.println("NEG: " + getNegation(head, men.dependency));
    System.out.println("MODAL: " + getModal(head, men.dependency));*/

    out.write(doc.conllDoc.documentID + ",");
    out.write(men.headWord.get(TokenBeginAnnotation.class) + ",");
    out.write(men.toString().replaceAll(",", "/") + ",");
    out.write(Mention.sentenceWordsToString(men).replaceAll(",", "/").trim() + ",");
    out.write(men.headString.replaceAll(",", "/") + ",");
    out.write(men.nerString + ",");
    out.write(men.animacy + ",");
    out.write(String.valueOf(getPerson(men.person)) + ",");
    out.write(men.number + ",");
    out.write(getPosition(men) + ",");
    out.write(getRelation(head, men.dependency) + ",");
    out.write(getQuantification(head, men) + ",");
    out.write(getModifiers(head, men) + ",");
    out.write(getNegation(head, men.dependency) + ",");
    out.write(getModal(head, men.dependency) + ",");
    out.write(getReportEmbedding(head, men.dependency) + ",");
    out.write(String.valueOf(getCoordination(head, men.dependency)));
    out.write("\n");
  }

  private static int[] getEntityVector(TreeSet<Mention> entity){

    int[] lifespan_vector = new int[entity.size()-1];
    int prev_token_index = 0;
    int start_position = 0;
    for(Mention m : entity){     
      if(m.equals(entity.first())){
        prev_token_index = m.headWord.get(TokenBeginAnnotation.class);
      } else {
        lifespan_vector[start_position++] = m.headWord.get(TokenBeginAnnotation.class) - prev_token_index;      
      }      
    }

    /*System.out.print("( ");
    for(int i = 0; i < lifespan_vector.length; i++){
      System.out.print(lifespan_vector[i]+" ");
    }
    System.out.print(")");
    System.out.println();*/
    return lifespan_vector;
  }

  private static String getEntityClass(TreeSet<Mention> entity){
    int sent_num = entity.first().headWord.sentIndex(); 
    for(Mention men : entity){
      if(men.headWord.sentIndex() != sent_num) return "long";
    }   
    return "short";
  }

  private static double getEntityScore(int[] lifespan_vector){
    double score = 0.0;
    for(int i = 0; i < lifespan_vector.length; i++){
      score += (double) (i+1)/lifespan_vector[i];
    }

    return score;
  }

  private static double getMaxScore(ArrayList<TreeSet<Mention>> entities){
    double max = 0.0;   
    for(TreeSet<Mention> entity : entities){
      double score = getEntityScore(getEntityVector(entity));
      if(score > max) max = score;      
    }

    return max;
  }

  private static String getRelation(IndexedWord head, SemanticGraph dependency){
    if(dependency.getRoots().isEmpty()) return null;
    // root relation
    if(dependency.getFirstRoot().equals(head)) return "root";
    if(!dependency.vertexSet().contains(dependency.getParent(head))) return null;
    GrammaticalRelation relation = dependency.reln(dependency.getParent(head), head);

    // adjunct relations
    if(relation.toString().startsWith("prep") || relation == EnglishGrammaticalRelations.PREPOSITIONAL_OBJECT || relation == EnglishGrammaticalRelations.TEMPORAL_MODIFIER || relation == EnglishGrammaticalRelations.ADV_CLAUSE_MODIFIER || relation == EnglishGrammaticalRelations.ADVERBIAL_MODIFIER || relation == EnglishGrammaticalRelations.PREPOSITIONAL_COMPLEMENT) return "adjunct";

    // subject relations
    if(relation == EnglishGrammaticalRelations.NOMINAL_SUBJECT || relation == EnglishGrammaticalRelations.CLAUSAL_SUBJECT || relation == EnglishGrammaticalRelations.CONTROLLING_SUBJECT) return "subject";
    if(relation == EnglishGrammaticalRelations.NOMINAL_PASSIVE_SUBJECT || relation == EnglishGrammaticalRelations.CLAUSAL_PASSIVE_SUBJECT) return "subject";

    // verbal argument relations
    if(relation == EnglishGrammaticalRelations.ADJECTIVAL_COMPLEMENT || relation == EnglishGrammaticalRelations.ATTRIBUTIVE || relation == EnglishGrammaticalRelations.CLAUSAL_COMPLEMENT || relation == EnglishGrammaticalRelations.XCLAUSAL_COMPLEMENT || relation == EnglishGrammaticalRelations.AGENT || relation == EnglishGrammaticalRelations.DIRECT_OBJECT || relation == EnglishGrammaticalRelations.INDIRECT_OBJECT) return "verbArg";

    // noun argument relations
    if(relation == EnglishGrammaticalRelations.RELATIVE_CLAUSE_MODIFIER || relation == EnglishGrammaticalRelations.NOUN_COMPOUND_MODIFIER || relation == EnglishGrammaticalRelations.ADJECTIVAL_MODIFIER || relation == EnglishGrammaticalRelations.APPOSITIONAL_MODIFIER || relation == EnglishGrammaticalRelations.POSSESSION_MODIFIER) return "nounArg";

    return null;
  }

  private static int getCoordination(IndexedWord head, SemanticGraph dependency) {
    Set<GrammaticalRelation> relations = dependency.childRelns(head);
    for (GrammaticalRelation rel : relations) {
      if(rel.toString().startsWith("conj_")) {
        //System.out.println("COORD");
        return 1;
      }
    }

    Set<GrammaticalRelation> parent_relations = dependency.relns(head);
    for (GrammaticalRelation rel : parent_relations) {
      if(rel.toString().startsWith("conj_")) {
        //System.out.println("COORD");
        return 1;
      }
    }

    return 0;
  }

  private static int getModifiers(IndexedWord head, Mention mention){
    SemanticGraph dependency = mention.dependency;
    int count = 0;
    List<Pair<GrammaticalRelation, IndexedWord>> childPairs = dependency.childPairs(head);
    for(Pair<GrammaticalRelation, IndexedWord> childPair : childPairs) {
      GrammaticalRelation gr = childPair.first;
      IndexedWord word = childPair.second;
      if(gr == EnglishGrammaticalRelations.ADJECTIVAL_MODIFIER || gr == EnglishGrammaticalRelations.PARTICIPIAL_MODIFIER 
          || gr == EnglishGrammaticalRelations.RELATIVE_CLAUSE_MODIFIER || gr == EnglishGrammaticalRelations.INFINITIVAL_MODIFIER 
          || gr.toString().startsWith("prep_")) {
        count++;  
      }
      // add noun modifier when the mention isn't a NER
      if(mention.nerString.equals("O") && gr == EnglishGrammaticalRelations.NOUN_COMPOUND_MODIFIER) {
        count++;
      }

      // add possessive if not a personal determiner
      if(gr == EnglishGrammaticalRelations.POSSESSION_MODIFIER && !determiners.contains(word.lemma())) {
        count++;
      }
    }
    //System.out.println("# MOD: " + count);
    return count;
  }

  private static int getReportEmbedding(IndexedWord head, SemanticGraph dependency) {

    // check adverbial clause with marker "as"
    Collection<IndexedWord> siblings = dependency.getSiblings(head);
    for(IndexedWord sibling : siblings) {
      if(reportVerb.contains(sibling.lemma()) && dependency.hasParentWithReln(sibling,EnglishGrammaticalRelations.ADV_CLAUSE_MODIFIER)) {
        IndexedWord marker = dependency.getChildWithReln(sibling,EnglishGrammaticalRelations.MARKER);
        if (marker != null && marker.lemma().equals("as")) {
          //System.out.println("UNDER ATTITUDE VERB");          
          return 1;
        }
      }
    }

    // look at the path to root
    List<IndexedWord> path = dependency.getPathToRoot(head);
    if(path == null) return 0;
    Boolean isSubject = false;

    // if the node itself is a subject, we will not take into account its parent in the path
    if(dependency.hasParentWithReln(head,EnglishGrammaticalRelations.NOMINAL_SUBJECT)) isSubject = true;

    for(IndexedWord word : path) {
      if(!isSubject && (reportVerb.contains(word.lemma()) || reportNoun.contains(word.lemma()))) {
        //System.out.println("UNDER ATTITUDE VERB");
        return 1;
      }
      // check how to put isSubject
      if(dependency.hasParentWithReln(word,EnglishGrammaticalRelations.NOMINAL_SUBJECT)) isSubject = true;
      else isSubject = false;
    }

    return 0;
  }

  private static String getQuantification(IndexedWord head, Mention men){

    if(!men.nerString.equals("O")) return "definite";

    SemanticGraph dependency = men.dependency;
    List<IndexedWord> quant = dependency.getChildrenWithReln(head,EnglishGrammaticalRelations.DETERMINER);
    List<IndexedWord> poss = dependency.getChildrenWithReln(head,EnglishGrammaticalRelations.POSSESSION_MODIFIER);
    String det = "";
    if(!quant.isEmpty()) {
      det = quant.get(0).lemma();
      if(determiners.contains(det)) {
        return "definite";
      }
    }
    else if(!poss.isEmpty()) {
      return "definite";  
    }
    else {
      quant = dependency.getChildrenWithReln(head,EnglishGrammaticalRelations.NUMERIC_MODIFIER);
      if(quantifiers.contains(det) || !quant.isEmpty()) {
        return "quantified";
      }
    }
    return "indefinite";
  }

  private static int getNegation(IndexedWord head, SemanticGraph dependency) {
    // direct negation in a child
    Collection<IndexedWord> children = dependency.getChildren(head);
    for(IndexedWord child : children) {
      if(negations.contains(child.lemma())) return 1;
    }

    // or has a sibling
    Collection<IndexedWord> siblings = dependency.getSiblings(head);
    for(IndexedWord sibling : siblings) {
      if(negations.contains(sibling.lemma()) && !dependency.hasParentWithReln(head,EnglishGrammaticalRelations.NOMINAL_SUBJECT)) return 1;
    }
    // check the parent
    List<Pair<GrammaticalRelation,IndexedWord>> parentPairs = dependency.parentPairs(head);
    if (!parentPairs.isEmpty()) {
      Pair<GrammaticalRelation,IndexedWord> parentPair = parentPairs.get(0);
      GrammaticalRelation gr = parentPair.first; 
      // check negative prepositions
      if(neg_relations.contains(gr.toString())) return 1;
    }

    return 0;    
  }

  private static int getModal(IndexedWord head, SemanticGraph dependency) {
    // direct modal in a child
    Collection<IndexedWord> children = dependency.getChildren(head);
    for(IndexedWord child : children) {
      if(modals.contains(child.lemma())) return 1;
    }

    // check the parent
    IndexedWord parent = dependency.getParent(head);
    if (parent != null) {
      if(modals.contains(parent.lemma())) return 1;
      // check the children of the parent (that is needed for modal auxiliaries)
      IndexedWord child = dependency.getChildWithReln(parent,EnglishGrammaticalRelations.AUX_MODIFIER);
      if(!dependency.hasParentWithReln(head,EnglishGrammaticalRelations.NOMINAL_SUBJECT) && child != null && modals.contains(child.lemma())) return 1;      
    }

    // look at the path to root
    List<IndexedWord> path = dependency.getPathToRoot(head);
    if(path == null) return 0;
    for(IndexedWord word : path) {
      if(modals.contains(word.lemma())) return 1;
    }

    return 0;
  }

  private static int getPerson(Person person){
    if(person.equals(Person.I) || person.equals(Person.WE)){
      return 1;
    } else if(person.equals(Person.YOU)){
      return 2;
    } else if(person.equals(Person.UNKNOWN)){
      return 0;
    } else {
      return 3;
    }
  }

  private static String getPosition(Mention men) {
    int index = men.headIndex;
    int size = men.sentenceWords.size();
    if(index == 0) {
      return "first";
    }
    else if (index == size -1) {
      return "last";
    }
    else {
      if(index > 0 && index < size/3) {
        return "begin";
      }
      else if (index >= size/3 && index < 2 * size/3) {
        return "middle";
      }
      else if (index >= 2 * size/3 && index < size -1) {
        return "end";
      }
    }
    return null;
  }

  // Set index for each token and sentence in the document 
  public static void setTokenIndices(Document doc){
    int token_index = 0;
    //int sent_index = -1;
    for(CoreMap sent : doc.annotation.get(SentencesAnnotation.class)){
      //sent_index++;
      for(CoreLabel token : sent.get(TokensAnnotation.class)){
        token.set(TokenBeginAnnotation.class, token_index++);
        //token.set(SentenceIndexAnnotation.class, sent_index);
      }
    }
  }

  private static class MentionComparator implements Comparator<Mention>{
    public int compare(Mention m1, Mention m2){
      return m1.headWord.get(TokenBeginAnnotation.class) - m2.headWord.get(TokenBeginAnnotation.class);
    }
  }

  public static void main(String[] args) throws Exception {
    Properties props = new Properties();
    props.setProperty("dcoref.conll2011", args[0]);

    BufferedWriter out = new BufferedWriter(new FileWriter(args[1]));

    out.write("Label,DocID,MentionID,Mention,Sentence,Head,NER,Animacy,Person,Number,SentencePosition,Relation,Quantifier,Modifiers,Negation,Modal,AttitudeVerb,Coordination");
    out.write("\n");

    SieveCoreferenceSystem corefSystem = new SieveCoreferenceSystem(props);
    Dictionaries dict = new Dictionaries(props);
    MentionExtractor mentionExtractor = new CoNLLMentionExtractor(dict, props, corefSystem.new Semantics(dict));

    Document document;
    while((document = mentionExtractor.nextDoc()) != null){
      setTokenIndices(document);
      document.extractGoldCorefClusters();
      Map<Integer, CorefCluster> entities = document.goldCorefClusters;

      ArrayList<TreeSet<Mention>> ordered_entities = new ArrayList<TreeSet<Mention>>();     
      for(CorefCluster entity : entities.values()){
        TreeSet<Mention> ordered_mentions = new TreeSet<Mention>(new MentionComparator());
        ordered_mentions.addAll(entity.getCorefMentions());
        ordered_entities.add(ordered_mentions);
      }
      //double max_score = getMaxScore(ordered_entities);

      for(TreeSet<Mention> ordered_mentions : ordered_entities){

        // First mentions
        Mention mention = ordered_mentions.first();

        // All coref mentions
        //for(Mention mention : ordered_mentions){

        // Ignore verbal mentions
        if(mention.headWord.tag().startsWith("V")) continue;

        IndexedWord head = mention.dependency.getNodeByIndexSafe(mention.headWord.index());
        if(head == null) continue;
        //double score = getEntityScore(getEntityVector(ordered_mentions)) / max_score;
        //out.write(String.format("%.4f", score) + ",");    
        //out.write(getEntityClass(ordered_mentions) + ",");
        out.write("1,");
        getFeatures(out, document, mention, head);
        //}
      }

      // Add singletons     
      ArrayList<CoreLabel> gold_heads = new ArrayList<CoreLabel>();
      for(Mention gold_men : document.allGoldMentions.values()){
        gold_heads.add(gold_men.headWord);
      }      
      for(Mention predicted_men : document.allPredictedMentions.values()){
        SemanticGraph dep = predicted_men.dependency;
        IndexedWord head = dep.getNodeByIndexSafe(predicted_men.headWord.index());
        if(head == null || !dep.vertexSet().contains(head)) continue;

        if(!predicted_men.isVerb && !gold_heads.contains(predicted_men.headWord)){
          //out.write("0.0,");
          //out.write("singleton,");
          out.write("0,");
          getFeatures(out, document, predicted_men, head);         
        }
      }

    }
    out.close();
  }

}
