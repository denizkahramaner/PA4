package cs224n.coref;

import java.util.*;
public class GenderSpeakerNumberFilter implements Filter{
	
	public Map<Mention,HashSet<Mention>> filter(Map<Mention,HashSet<Mention>> mentions,Document doc){
		
		for(Mention m : mentions.keySet()){
			Iterator<Mention> it = mentions.get(m).iterator();
			while(it.hasNext()){
				Mention n = it.next();
			
				if(m.gloss().contains("Cole") || n.gloss().contains("Cole")){
					int counter =0;
					
				}
				if(Pronoun.isSomePronoun(m.headWord()) && Pronoun.isSomePronoun(n.headWord())){
					Pronoun p1 = Pronoun.valueOrNull(m.headWord());
					Pronoun p2 = Pronoun.valueOrNull(n.headWord());
					//boolean match = false;
					if(p1!=null && p2!=null){
						if(!(p1.gender.equals(p2.gender) && p1.speaker.equals(p2.speaker) && p1.plural == p2.plural )){
							it.remove();
							continue;
						}
					}
				}

				if(Name.isName(m.headWord()) && Pronoun.isSomePronoun(n.headWord())){
					Name n1 = Name.get(m.headWord());
					Pronoun p = Pronoun.valueOrNull(n.headWord());
					//boolean match = false;
					if(n1!=null && p!=null){
						if(!(n1.mostLikelyGender().equals( p.gender) ) ){
							it.remove();
							continue;
						}
					}
				}

				if(Name.isName(n.headWord()) && Pronoun.isSomePronoun(m.headWord())){
					Name n1 = Name.get(n.headWord());
					Pronoun p = Pronoun.valueOrNull(m.headWord());
					//boolean match = false;
					if(n1!=null && p!=null){
						if(!(n1.mostLikelyGender().equals( p.gender)  ) ){
							it.remove();
							continue;
						}
					}
				}
				
				if(Name.isName(m.headWord()) && Name.isName(n.headWord())){
					Name n1 = Name.get(m.headWord());
					Name n2 = Name.get(n.headWord());
					//boolean match = false;
					if(n1!=null && n2!=null){
						if(!(n1.gender.equals( n2.gender) )){
							it.remove();
							continue;
						}
					}
				}
				
				if(m.headToken().isNoun() && (n.headToken().isNoun()|| n.headToken().isQuoted())){
					String ner1= m.headToken().nerTag();
					String ner2 =n.headToken().nerTag();

					if(!m.headToken().nerTag().equals(n.headToken().nerTag()) || !(m.headToken().isPluralNoun() == n.headToken().isPluralNoun())){
						it.remove();
						continue;
					}
										
				}
				
				if(m.headToken().isNoun() && Pronoun.isSomePronoun(n.headWord())){
					Pronoun p = Pronoun.valueOrNull(n.headWord());				
					if(p!=null && (!m.headToken().isPluralNoun() == p.plural /*|| (p.gender!=Gender.NEUTRAL && m.headToken().isProperNoun() )*/)){
						it.remove();
						continue;
					}
										
				}
				
				if(n.headToken().isNoun() && Pronoun.isSomePronoun(m.headWord())){
					Pronoun p = Pronoun.valueOrNull(m.headWord());
					//String tag = n.headToken().nerTag();
					if(p!=null &&(!n.headToken().isPluralNoun() == p.plural /*|| (p.gender!=Gender.NEUTRAL && (n.headToken().isProperNoun() ) )*/ )){
						it.remove();
						continue;
					}
										
				}
				
				
				
			}
			
		}
		
		return mentions;
	}
	
}