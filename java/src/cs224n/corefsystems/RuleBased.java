package cs224n.corefsystems;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import cs224n.coref.ClusteredMention;
import cs224n.coref.Document;
import cs224n.coref.Entity;
import cs224n.coref.Mention;
import cs224n.util.Counter;
import cs224n.util.Pair;
import cs224n.coref.*;

public class RuleBased implements CoreferenceSystem {

	private List<RuleSet> passes;
	private int dummycnt;
	public void setUpRules(){
		//Set up the passes for this Rule Base Coreference system.
		// Each pass is represented by a ruleset which essentially has some rules to generate candidate coreferences and 
		// some filters to remove some.
		// Ideally the pass setting up should not be "manual" but should be parametrized as commandline parameters
		// But that is not something I wanna get into now.
		passes = new ArrayList<RuleSet>();
		
		List<Rule> rulesFirstPass = new ArrayList<Rule>();
		rulesFirstPass.add(new HeadMatchingRule());
		rulesFirstPass.add(new HeadEditDistanceRule());
		
		
		List<Filter> filtersFirstPass = new ArrayList<Filter>();
		RuleSet firstPass = new RuleSet(rulesFirstPass,filtersFirstPass);
		passes.add(firstPass);

		List<Rule> rulesSecondPass = new ArrayList<Rule>();
		rulesSecondPass.add(new NaivePronounRule());
		rulesSecondPass.add(new TokenMatchRule());
		//rulesSecondPass.add(new HobbsCandidatesRule());
		List<Filter> filtersSecondPass = new ArrayList<Filter>();
		filtersSecondPass.add(new DistanceFilter());
		filtersSecondPass.add(new GenderSpeakerNumberFilter());
		RuleSet secondPass = new RuleSet(rulesSecondPass,filtersSecondPass);
		passes.add(secondPass);
		
	}
	@Override
	public void train(Collection<Pair<Document, List<Entity>>> trainingData) {
		// TODO Auto-generated method stub
		this.setUpRules();
		this.dummycnt =0;
	}

	
	@Override
	public List<ClusteredMention> runCoreference(Document doc) {
		// TODO Auto-generated method stub
		List<ClusteredMention> mentions = new ArrayList<ClusteredMention>();
		//mentions= new AllSingleton().runCoreference(doc);
		Map<Mention,HashSet<Mention>> clusters = new HashMap<Mention, HashSet<Mention>>();
		
		for(RuleSet r : this.passes){
			Map<Mention,HashSet<Mention>> current = r.apply(doc);
			if(this.dummycnt ==0 ){
				//System.out.println(current.toString());
			
			}
			for(Mention m: current.keySet()){
				if(clusters.containsKey(m)){
					clusters.get(m).addAll(current.get(m));
				}
				else{
					clusters.put(m, current.get(m));
				}
			}

		}
		if(this.dummycnt == 0){
			//System.out.println(clusters.toString());
		}
		this.dummycnt = 1;
		
		//HashSet<Integer> hash = new HashSet<Integer>();
	    Map<Integer,Entity> hash = new HashMap<Integer,Entity>();

		
		for(Mention m: doc.getMentions()){
			
			if(hash.containsKey(new Integer(m.hashCode()))){
				if(clusters.get(m)!=null){
					for(Mention n :clusters.get(m)){
						if(hash.containsKey(new Integer(n.hashCode()))){
							continue;
						}
		            	mentions.add(n.markCoreferent(hash.get(new Integer(m.hashCode()))));
		            	hash.put(new Integer(n.hashCode()),hash.get(new Integer(m.hashCode())));
		        	
		        	}
				}
			}
			else{
				boolean isAnyEncountered = false;
				Mention encountered = null;
				if(clusters.get(m)!=null){
			        for(Mention n :clusters.get(m)){
						if(hash.containsKey(new Integer(n.hashCode()))){
							isAnyEncountered = true;
							encountered = n;
							break;
						}
			        	
			        }
		          }
	           if(isAnyEncountered){
	        		if(hash.containsKey(new Integer(m.hashCode()))){
						continue;
					}
	            	mentions.add(m.markCoreferent(hash.get(new Integer(encountered.hashCode()))));
	            	hash.put(new Integer(m.hashCode()),hash.get(new Integer(encountered.hashCode())));
			        if(clusters.get(m)!=null){
			        	for(Mention n :clusters.get(m)){
			        		if(hash.containsKey(new Integer(n.hashCode()))){
								continue;
							}

			        		mentions.add(n.markCoreferent(hash.get(new Integer(encountered.hashCode()))));
			        		hash.put(new Integer(n.hashCode()),hash.get(new Integer(encountered.hashCode())));

			        	}
			        }

	           }
	           else{
	        	if(hash.containsKey(new Integer(m.hashCode()))){
						continue;
				}
			     ClusteredMention newCluster = m.markSingleton();
			     mentions.add(newCluster);
		         hash.put(new Integer(m.hashCode()),newCluster.entity);
		         if(clusters.get(m)!=null){
			        for(Mention n :clusters.get(m)){
		        		if(hash.containsKey(new Integer(n.hashCode()))){
							continue;
						}
		            	mentions.add(n.markCoreferent(hash.get(new Integer(m.hashCode()))));
		            	hash.put(new Integer(n.hashCode()),hash.get(new Integer(m.hashCode())));

			        }
		         }

	           }

			}
		}
			
	
		return mentions;
	}

}
