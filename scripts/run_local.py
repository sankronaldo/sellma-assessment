"""
Self-contained local experiment runner.

Uses synthetic Swedish/English sentence pairs (no network download needed for data).
Loads model from ./models/Qwen2.5-0.5B-Instruct (already downloaded via git clone).

Run with:
  python -I scripts/run_local.py

Expected runtime on RTX 2050: ~15-25 minutes.
"""

import copy
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

MODEL_PATH = str(ROOT / "models" / "Qwen2.5-0.5B-Instruct")
RESULTS_DIR = ROOT / "results"
CONFIG_DIR = ROOT / "configs"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CONFIG_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Synthetic dataset (50 dev + 100 test, disjoint, no download needed)
# ---------------------------------------------------------------------------

SV_SENTENCES_DEV = [
    "Sverige är ett land i norra Europa med cirka tio miljoner invånare.",
    "Stockholm är Sveriges huvudstad och landets största stad.",
    "Det svenska klimatet är varierat med kalla vintrar och varma somrar.",
    "Den svenska grundlagen garanterar yttrande- och pressfrihet.",
    "Sverige har en lång tradition av demokrati och välfärdsstat.",
    "Det svenska utbildningssystemet är välkänt för sin höga kvalitet.",
    "Många svenska företag är världsledande inom teknik och innovation.",
    "Den svenska naturen är känd för sina skogar, sjöar och fjord.",
    "Det svenska köket inkluderar rätter som köttbullar, smörgåsbord och surströmming.",
    "Sverige deltar aktivt i internationella organisationer som FN och EU.",
    "Det svenska språket tillhör den nordgermanska språkgruppen.",
    "Midsommar är en av Sveriges viktigaste traditioner och firas i juni.",
    "Den kungliga familjen spelar en ceremoniel roll i det svenska samhället.",
    "Sverige är känt för sin höga levnadsstandard och sociala trygghet.",
    "Det svenska rättssystemet bygger på rättsstatliga principer och mänskliga rättigheter.",
    "Svenska medborgare har fri tillgång till hälso- och sjukvård.",
    "Litteratur och konst har en stark ställning i svensk kultur.",
    "Sverige exporterar bland annat stål, fordon och läkemedel.",
    "Det svenska transportnätet inkluderar järnvägar, motorvägar och flyg.",
    "Folkbildning och bibliotek är viktiga institutioner i det svenska samhället.",
    "Sverige har en stark tradition av jämlikhet mellan könen.",
    "Naturreservat och nationalparker skyddar den svenska biologiska mångfalden.",
    "Det svenska pensionssystemet bygger på livslång arbetsinsats.",
    "Sverige rankas högt i internationella index för pressfriheten.",
    "Det svenska civilsamhället är starkt med många ideella organisationer.",
    "Musik och film från Sverige har nått internationell framgång.",
    "Den svenska arbetsrätten ger anställda starka skydd och rättigheter.",
    "Sverige satsar på förnybar energi som vind- och vattenkraft.",
    "Det svenska samhällets värderingar inkluderar öppenhet och tolerans.",
    "Böcker och tidningar är populära informationskällor i Sverige.",
    "Det nordiska samarbetet mellan Sverige, Norge, Danmark, Finland och Island är unikt.",
    "Sverige är en konstitutionell monarki med en folkvald riksdag.",
    "Det svenska rättsväsendet är känt för sin opartiskhet.",
    "Högskoleutbildning i Sverige är avgiftsfri för svenska medborgare.",
    "Det svenska museet Nationalmuseum hyser en stor konstsamling.",
    "Den svenska modellen kombinerar marknadsekonomi med generösa sociala förmåner.",
    "Sverige spelar en aktiv roll i det europeiska samarbetet.",
    "Det svenska skattesystemet finansierar den offentliga sektorn.",
    "Städerna i Sverige erbjuder god kollektivtrafik och cykelinfrastruktur.",
    "Det svenska fisket är en viktig del av landets ekonomi och kultur.",
    "Sverige har tagit emot många flyktingar och invandrare under åren.",
    "Det svenska systemet för barnbidrag och föräldraförsäkring är välkänt.",
    "Forskning och utveckling prioriteras högt i Sverige.",
    "Den svenska skolan är icke-konfessionell och avgiftsfri.",
    "Sverige deltar i internationella fredsmissioner och humanitärt bistånd.",
    "Det svenska jordbruket moderniserades kraftigt under 1900-talet.",
    "Sverige har en stark tradition av hantverk och design.",
    "Det svenska banksystemet är stabilt och välutvecklat.",
    "Den svenska befolkningen är välutbildad och har hög medellivslängd.",
    "Sverige samarbetar med andra länder i klimat- och miljöfrågor.",
]

SV_SENTENCES_TEST = [
    "Huset på kullen är stort och vitt med ett rött tak.",
    "Katten sover på den mjuka soffan i vardagsrummet.",
    "Barnen leker glatt i parken under en solig dag.",
    "Himlen är klarblå och solen skiner starkt idag.",
    "Boken på bordet handlar om historiska händelser i Europa.",
    "Fågeln sjunger vackert i det gamla trädet utanför fönstret.",
    "Vattnet i den lilla sjön är kallt och klart som glas.",
    "Hunden springer längs stranden och skäller på vågorna.",
    "Läraren förklarar matematik för de uppmärksamma eleverna.",
    "Regnet faller mjukt utanför det öppna fönstret.",
    "Staden är full av glada människor på en vacker lördagsmorgon.",
    "Bilen stannade sakta vid de röda trafikljuset vid korsningen.",
    "Blomman öppnar sina kronblad långsamt i det varma solskenet.",
    "Biblioteket har tusentals böcker på alla möjliga ämnen.",
    "Floden rinner stilla mot havet i söder.",
    "Skolan börjar punktligt klockan åtta varje vardag.",
    "Fiskarna simmar i stim i det klara vattnet.",
    "Vinden blåser starkt från norr och det är kallt.",
    "Mannen läser morgontidningen vid köksbordet med en kopp kaffe.",
    "Flickan skriver ett brev till sin brevvän i Finland.",
    "Trädgården är full av färgglada blommor på sommaren.",
    "Pojken lär sig spela gitarr varje kväll efter skolan.",
    "Det gamla huset vid sjön har ett grönmålat staket.",
    "Familjen äter middag tillsammans varje söndag kväll.",
    "Tåget anlände till stationen exakt i tid.",
    "Snön täckte hela byn med ett vitt täcke över natten.",
    "Läraren berättar en spännande historia för klassen.",
    "Solen går ner bakom bergen och himlen blir orange.",
    "Barnet leker med sina leksaker på golvet i rummet.",
    "Kaféet serverar gott kaffe och nybakat bröd varje morgon.",
    "Trädgårdsmästaren planterar blommor längs gången.",
    "Båten seglar sakta ut på det lugna havet.",
    "Det gamla tornet syns från hela staden.",
    "Fåglarna bygger bon i de höga träden.",
    "Vägen slingrar sig upp för berget mot byn.",
    "Hönorna pickar korn i den lilla gårdsplanen.",
    "Klockorna ringer vackert från den gamla kyrkan.",
    "Fisken hoppar upp ur vattnet i den klara bäcken.",
    "Hjorten springer snabbt genom den täta skogen.",
    "Vintern är kall men vacker med snö och is.",
    "Åkrarna är gröna och fulla av säd på sommaren.",
    "Hamnen är livlig med båtar och sjöfolk.",
    "Stenbron är gammal men fortfarande stabil.",
    "Bagaren bakar färskt bröd tidigt varje morgon.",
    "Byns skola har tjugo elever i de lägre klasserna.",
    "Torpet vid sjön används som sommarstuga.",
    "Korna betar lugnt på den gröna ängen.",
    "Soldaterna marscherar i takt på torget.",
    "Sjuksköterskan tar hand om patienterna på avdelningen.",
    "Handlaren säljer grönsaker och frukt på torget.",
    "Bonden skördar säden med en modern skördetröska.",
    "Prästen håller gudstjänst i den fullsatta kyrkan.",
    "Fiskarens båt ligger förtöjd vid bryggan.",
    "Skomakaren lagar skor i sin lilla verkstad.",
    "Muraren bygger en mur av sten och kalk.",
    "Smeden arbetar vid ässjan och smidar järn.",
    "Hjulet på vagnen gick sönder på den ojämna vägen.",
    "Marknadens stånd var fulla av varor och besökare.",
    "Hantverkarna byggde huset med stort kunnande.",
    "Bonden sår frön i den välgödda åkern.",
    "Eken vid bäcken är hundra år gammal.",
    "Älgen betar lugnt i den dimhöljda skogen.",
    "Korpen sitter på staketet och kraxar.",
    "Björnen sover i sitt ide under hela vintern.",
    "Råven smyger fram i skymningen.",
    "Ugglans rop hörs i den mörka natten.",
    "Mossen lyser rött av lingon och hjortron.",
    "Himlen pryds av norrsken på vinternätterna.",
    "Laxen hoppar upp för forsarna.",
    "Hästarna galopperar fritt på den öppna slätten.",
    "Bondgårdens djur vaknar tidigt om morgonen.",
    "Bröderna delar likadant på arbetet och vinsten.",
    "Bysmeden var känd för sin styrka och skicklighet.",
    "Bonden önskar god skörd för det kommande året.",
    "Torparen arbetar hårt från gryning till kväll.",
    "Trädet vid vägen ger skugga åt de trötta vandrarna.",
    "Kvarnen maler säden till mjöl.",
    "Skeppet med vikingar seglade mot okända länder.",
    "Runan på stenen är svår att tyda.",
    "Kung Olof regerade med klokhet och rättvisa.",
    "Handelsmannen reste långa vägar för att sälja sina varor.",
    "Prästgårdens trädgård är full av äppelträd.",
    "Ättestupan är ett gammalt och omtvistat fenomen.",
    "Nordstjärnan lyser klart på den mörka himlen.",
    "Vikingatiden präglade Skandinavien under flera hundra år.",
    "Tingsstämman var viktig för att lösa tvister i byn.",
    "Sockenstämman beslutade om fattigvårdens organisation.",
    "Socknen hade en gemensam fäbod uppe i skogen.",
    "Ryttaren ledde sina hästar till vattenståndet.",
    "Torpets närmaste grannar bodde en timmes gångväg bort.",
    "Bonden satte igen stängselstolparna som stormen blåst ned.",
    "Årstidernas växling präglar livet på landsbygden.",
    "Midsommardansen runt stången är en gammal sed.",
    "Tomten vaktar gårdens djur och välgång.",
    "Vintersolståndet firas med ljus och glädje.",
    "Påskhögtiden välkomnas med ägg, blommor och fjäder.",
    "Lucia med sin ljusbärande procession upplyser mörkret.",
    "Allhelgonadagen hedrar de döda med ljus på gravarna.",
    "Skördefesten på hösten tackar för årets gröda.",
]

EN_SENTENCES_DEV = [
    "Sweden is a country in northern Europe with approximately ten million inhabitants.",
    "Stockholm is the capital and the largest city of Sweden.",
    "The Swedish climate is varied with cold winters and warm summers.",
    "The Swedish constitution guarantees freedom of speech and of the press.",
    "Sweden has a long tradition of democracy and the welfare state.",
    "The Swedish education system is well known for its high quality.",
    "Many Swedish companies are world leaders in technology and innovation.",
    "Swedish nature is known for its forests, lakes, and fjords.",
    "Swedish cuisine includes dishes like meatballs, smörgåsbord, and fermented herring.",
    "Sweden actively participates in international organizations such as the UN and EU.",
    "The Swedish language belongs to the North Germanic language group.",
    "Midsummer is one of Sweden's most important traditions, celebrated in June.",
    "The royal family plays a ceremonial role in Swedish society.",
    "Sweden is known for its high standard of living and social security.",
    "The Swedish legal system is based on the rule of law and human rights.",
    "Swedish citizens have free access to health and medical care.",
    "Literature and art have a strong position in Swedish culture.",
    "Sweden exports steel, vehicles, and pharmaceuticals, among other goods.",
    "The Swedish transport network includes railways, highways, and aviation.",
    "Public education and libraries are important institutions in Swedish society.",
    "Sweden has a strong tradition of gender equality.",
    "Nature reserves and national parks protect Sweden's biological diversity.",
    "The Swedish pension system is based on lifelong work contributions.",
    "Sweden ranks highly in international indices for press freedom.",
    "Swedish civil society is strong, with many non-profit organizations.",
    "Music and film from Sweden have achieved international success.",
    "Swedish labour law gives employees strong protections and rights.",
    "Sweden invests in renewable energy such as wind and hydropower.",
    "Swedish societal values include openness and tolerance.",
    "Books and newspapers are popular sources of information in Sweden.",
    "Nordic cooperation between Sweden, Norway, Denmark, Finland and Iceland is unique.",
    "Sweden is a constitutional monarchy with an elected parliament.",
    "The Swedish judiciary is known for its impartiality.",
    "Higher education in Sweden is free of charge for Swedish citizens.",
    "The Swedish museum Nationalmuseum holds a large art collection.",
    "The Swedish model combines a market economy with generous social benefits.",
    "Sweden plays an active role in European cooperation.",
    "The Swedish tax system finances the public sector.",
    "Cities in Sweden offer good public transport and cycling infrastructure.",
    "Swedish fishing is an important part of the country's economy and culture.",
    "Sweden has received many refugees and immigrants over the years.",
    "The Swedish system of child allowances and parental leave is well known.",
    "Research and development are given high priority in Sweden.",
    "Swedish schools are non-denominational and free of charge.",
    "Sweden participates in international peacekeeping missions and humanitarian aid.",
    "Swedish agriculture was greatly modernized during the twentieth century.",
    "Sweden has a strong tradition of crafts and design.",
    "The Swedish banking system is stable and well developed.",
    "The Swedish population is well educated and has a high life expectancy.",
    "Sweden cooperates with other countries on climate and environmental issues.",
]

EN_SENTENCES_TEST = [
    "The house on the hill is large and white with a red roof.",
    "The cat is sleeping on the soft sofa in the living room.",
    "The children are playing happily in the park on a sunny day.",
    "The sky is clear blue and the sun is shining brightly today.",
    "The book on the table is about historical events in Europe.",
    "The bird is singing beautifully in the old tree outside the window.",
    "The water in the small lake is cold and clear as glass.",
    "The dog runs along the beach and barks at the waves.",
    "The teacher explains mathematics to the attentive students.",
    "The rain falls softly outside the open window.",
    "The city is full of cheerful people on a beautiful Saturday morning.",
    "The car stopped slowly at the red traffic light at the intersection.",
    "The flower opens its petals slowly in the warm sunlight.",
    "The library has thousands of books on all possible subjects.",
    "The river flows quietly toward the sea in the south.",
    "School starts punctually at eight o'clock every weekday.",
    "The fish swim in a school in the clear water.",
    "The wind blows strongly from the north and it is cold.",
    "The man reads the morning newspaper at the kitchen table with a cup of coffee.",
    "The girl writes a letter to her pen friend in Finland.",
    "The garden is full of colourful flowers in summer.",
    "The boy learns to play guitar every evening after school.",
    "The old house by the lake has a green painted fence.",
    "The family eats dinner together every Sunday evening.",
    "The train arrived at the station exactly on time.",
    "The snow covered the whole village with a white blanket overnight.",
    "The teacher tells an exciting story to the class.",
    "The sun sets behind the mountains and the sky turns orange.",
    "The child plays with its toys on the floor in the room.",
    "The café serves good coffee and freshly baked bread every morning.",
    "The gardener plants flowers along the path.",
    "The boat sails slowly out onto the calm sea.",
    "The old tower can be seen from the whole city.",
    "The birds build nests in the tall trees.",
    "The road winds up the mountain toward the village.",
    "The hens peck grain in the small farmyard.",
    "The bells ring beautifully from the old church.",
    "The fish jumps out of the water in the clear stream.",
    "The deer runs quickly through the dense forest.",
    "Winter is cold but beautiful with snow and ice.",
    "The fields are green and full of grain in summer.",
    "The harbour is busy with boats and sailors.",
    "The stone bridge is old but still sturdy.",
    "The baker bakes fresh bread early every morning.",
    "The village school has twenty students in the lower grades.",
    "The cottage by the lake is used as a summer house.",
    "The cows graze peacefully on the green meadow.",
    "The soldiers march in step in the square.",
    "The nurse takes care of patients on the ward.",
    "The merchant sells vegetables and fruit at the market.",
    "The farmer harvests the grain with a modern combine harvester.",
    "The priest holds a service in the full church.",
    "The fisherman's boat is moored at the jetty.",
    "The shoemaker repairs shoes in his small workshop.",
    "The mason builds a wall of stone and lime.",
    "The blacksmith works at the forge and shapes iron.",
    "The wheel on the wagon broke on the uneven road.",
    "The market stalls were full of goods and visitors.",
    "The craftsmen built the house with great skill.",
    "The farmer sows seeds in the well-fertilized field.",
    "The oak by the stream is a hundred years old.",
    "The moose grazes peacefully in the fog-covered forest.",
    "The raven sits on the fence and caws.",
    "The bear sleeps in its den throughout the winter.",
    "The fox sneaks forward in the twilight.",
    "The owl's call is heard in the dark night.",
    "The bog glows red with lingonberries and cloudberries.",
    "The sky is decorated with the northern lights on winter nights.",
    "The salmon leaps up the rapids.",
    "The horses gallop freely on the open plain.",
    "The farm animals wake up early in the morning.",
    "The brothers share the work and profit equally.",
    "The village blacksmith was known for his strength and skill.",
    "The farmer wishes for a good harvest in the coming year.",
    "The crofter works hard from dawn until evening.",
    "The tree by the road provides shade for the tired travellers.",
    "The mill grinds the grain into flour.",
    "The Viking ship sailed toward unknown lands.",
    "The rune on the stone is difficult to decipher.",
    "King Olof ruled with wisdom and justice.",
    "The merchant travelled long distances to sell his goods.",
    "The parsonage garden is full of apple trees.",
    "The cliff sacrifice is an old and contested phenomenon.",
    "The North Star shines brightly in the dark sky.",
    "The Viking Age shaped Scandinavia for several hundred years.",
    "The local assembly was important for resolving disputes in the village.",
    "The parish meeting decided on the organization of poor relief.",
    "The parish had a shared summer farm up in the forest.",
    "The rider led his horses to the watering place.",
    "The croft's nearest neighbours lived an hour's walk away.",
    "The farmer replaced the fence posts blown down by the storm.",
    "The changing of the seasons shapes life in the countryside.",
    "The midsummer dance around the maypole is an ancient custom.",
    "The tomte guards the farm's animals and well-being.",
    "The winter solstice is celebrated with lights and joy.",
    "Easter is welcomed with eggs, flowers, and feathers.",
    "Lucia with her light-bearing procession illuminates the darkness.",
    "All Saints' Day honours the dead with candles on graves.",
    "The harvest festival in autumn gives thanks for the year's crop.",
]

PPL_SENTENCES = [
    "Det svenska klimatet präglas av kalla vintrar och relativt varma somrar.",
    "Sverige är en konstitutionell monarki med en folkvald riksdag som lagstiftande organ.",
    "Stockholms innerstad är belägen på ett flertal öar och halvöar i Mälaren.",
    "Den svenska litteraturens historia sträcker sig tillbaka till medeltiden.",
    "Det nordiska samarbetet har bidragit till fred och välstånd i regionen.",
    "Sverige har en av världens högsta andel förnybara energikällor.",
    "Det svenska rättssystemet bygger på principen om allas likhet inför lagen.",
    "Folkrörelserna spelade en central roll i det svenska demokratiseringsarbetet.",
    "Den svenska välfärdsstaten bygger på universella sociala rättigheter.",
    "Forskning och högre utbildning prioriteras högt i den svenska statsbudgeten.",
    "Det svenska skattesystemet omfördelar inkomster och finansierar den offentliga sektorn.",
    "Sverige deltar aktivt i internationella klimatförhandlingar.",
    "Den svenska pressen åtnjuter ett starkt grundlagsskydd.",
    "Folkbildningen har haft stor betydelse för det svenska demokratiska samhällets framväxt.",
    "Det svenska jordbruket kännetecknas av hög mekanisering och effektivitet.",
    "Sverige exporterar kunskapsintensiva varor och tjänster till hela världen.",
    "Den offentliga hälso- och sjukvården finansieras till stor del av skattemedel.",
    "Den svenska skolan syftar till att ge alla elever likvärdiga möjligheter.",
    "Naturreservaten skyddar viktiga ekosystem och biologisk mångfald.",
    "Det svenska banksystemet regleras av Finansinspektionen.",
]


def build_prompts(sentences, lang, prefix=""):
    prompts = []
    for i, s in enumerate(sentences):
        if lang == "sv":
            prompt = f"Sammanfatta följande mening på svenska:\n\n{s}\n\nSvar:"
        else:
            prompt = f"Summarize the following sentence in English:\n\n{s}\n\nAnswer:"
        prompts.append({"id": f"{prefix}{lang}_{i}", "prompt": prompt, "source_text": s, "expected_lang": lang})
    return prompts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"Device: {device}, dtype: {dtype}")

    # Load model
    print("\nLoading model from local path...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=dtype, trust_remote_code=True
    )
    model.eval()
    if device == "cuda":
        model = model.to(device)
    print(f"Model loaded: {sum(p.numel() for p in model.parameters())/1e6:.0f}M params")

    # Pin revision
    rev = getattr(model.config, "_commit_hash", "git-clone-modelscope")
    with open(CONFIG_DIR / "model_config.json", "w") as f:
        json.dump({"model_id": "qwen/Qwen2.5-0.5B-Instruct", "revision": rev,
                   "source": "modelscope-git-clone", "dtype": str(dtype), "device": device}, f, indent=2)

    # Build datasets
    print("\nBuilding datasets...")
    rng = random.Random(SEED)
    dev_sv = build_prompts(SV_SENTENCES_DEV, "sv", prefix="dev_")
    dev_en = build_prompts(EN_SENTENCES_DEV, "en", prefix="dev_")
    test_sv = build_prompts(SV_SENTENCES_TEST, "sv", prefix="test_")
    test_en = build_prompts(EN_SENTENCES_TEST, "en", prefix="test_")

    # Leakage check
    dev_ids = {p["id"] for p in dev_sv + dev_en}
    test_ids = {p["id"] for p in test_sv + test_en}
    assert len(dev_ids & test_ids) == 0, "Dev/test overlap!"
    print(f"  dev_sv={len(dev_sv)}, dev_en={len(dev_en)}, test_sv={len(test_sv)}, test_en={len(test_en)}")

    # Save split config
    split_cfg = {
        "seed": SEED, "dev_sv_ids": [p["id"] for p in dev_sv],
        "test_sv_ids": [p["id"] for p in test_sv],
        "dev_en_ids": [p["id"] for p in dev_en],
        "test_en_ids": [p["id"] for p in test_en],
        "note": "Synthetic sentences; dev and test are disjoint by construction"
    }
    with open(CONFIG_DIR / "data_split.json", "w", encoding="utf-8") as f:
        json.dump(split_cfg, f, ensure_ascii=False, indent=2)

    # --- Generation helper ---
    def run_condition(prompts, cond_name, system_prompt=None, model_to_use=None):
        m = model_to_use if model_to_use is not None else model
        m.eval()
        records = []
        for p in prompts:
            if system_prompt:
                msgs = [{"role": "system", "content": system_prompt},
                        {"role": "user", "content": p["prompt"]}]
            else:
                msgs = [{"role": "user", "content": p["prompt"]}]
            text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
            enc = {k: v.to(device) for k, v in enc.items()}
            input_len = enc["input_ids"].shape[1]
            with torch.no_grad():
                out = m.generate(**enc, max_new_tokens=60, do_sample=False,
                                  pad_token_id=tokenizer.eos_token_id)
            resp = tokenizer.decode(out[0][input_len:], skip_special_tokens=True).strip()
            records.append({"id": p["id"], "condition": cond_name,
                             "prompt": p["prompt"], "source_text": p["source_text"],
                             "expected_lang": p["expected_lang"], "response": resp})
        out_dir = RESULTS_DIR / "test_sv"
        out_dir.mkdir(exist_ok=True)
        import jsonlines
        with jsonlines.open(str(out_dir / f"{cond_name}.jsonl"), "w") as w:
            w.write_all(records)
        return records

    def eval_records(records, expected_lang="sv"):
        from src.metrics import language_id_accuracy, proportion_ci
        r = language_id_accuracy(records, expected_lang)
        ci = proportion_ci(r["correct"], r["n"])
        return {"n": r["n"], "correct": r["correct"],
                "lang_id_accuracy": ci["mean"],
                "lang_id_ci_lower": ci["lower"], "lang_id_ci_upper": ci["upper"]}

    # -----------------------------------------------------------------------
    # Phase 1: Compute language direction on dev (feedback-based — declared)
    # -----------------------------------------------------------------------
    print("\n[DEV] Computing language direction (FEEDBACK-BASED)...")
    from src.edit_utils import compute_language_direction, layer_direction_scores, apply_direction_ablation
    from src.edit_utils import apply_random_direction_edit, apply_sign_flipped_edit, count_edited_params

    sv_texts = [p["prompt"] for p in dev_sv]
    en_texts = [p["prompt"] for p in dev_en]
    directions = compute_language_direction(model, tokenizer, sv_texts, en_texts, device=device)
    scores = layer_direction_scores(directions, model)
    print(f"  Top-5 layers: {scores[:5]}")

    # Tune alpha on dev
    best_alpha, best_acc = 1.0, -1.0
    (RESULTS_DIR / "dev_tune").mkdir(exist_ok=True)
    for alpha in [0.5, 1.0, 2.0]:
        edited = apply_direction_ablation(
            model, directions, target_layers=[scores[0][0]], alpha=alpha, n_columns=64)
        recs = run_condition(dev_sv[:20], f"dev_alpha_{alpha}", model_to_use=edited)
        acc = eval_records(recs, "sv")["lang_id_accuracy"]
        print(f"  alpha={alpha:.1f}: dev lang-ID={acc:.2%}")
        if acc > best_acc:
            best_acc, best_alpha = acc, alpha

    top_layer = scores[0][0]
    target_layers = [top_layer]
    print(f"\n  Chosen: layer={top_layer}, alpha={best_alpha} (dev acc={best_acc:.2%})")

    edit_cfg = {"target_layers": target_layers, "alpha": best_alpha, "n_cols": 64,
                "seed": SEED, "layer_scores": scores[:10]}
    with open(CONFIG_DIR / "edit_config.json", "w") as f:
        json.dump(edit_cfg, f, indent=2)

    # -----------------------------------------------------------------------
    # Phase 2: Weight-patch helper (avoid storing multiple full model copies)
    # -----------------------------------------------------------------------
    # Instead of deepcopy, we save only the edited parameter tensors as CPU
    # snapshots and restore the base weights after each condition.
    # This keeps GPU memory at 1 model at a time.

    def save_base_weights(m, layer_indices):
        """Save only the down_proj weights for target layers (CPU copy)."""
        saved = {}
        for li in layer_indices:
            w = m.model.layers[li].mlp.down_proj.weight.data
            saved[li] = w.detach().cpu().clone()
        return saved

    def restore_weights(m, saved):
        """Restore saved weights back to the model."""
        with torch.no_grad():
            for li, w_cpu in saved.items():
                m.model.layers[li].mlp.down_proj.weight.data.copy_(w_cpu.to(m.device))

    # Save base weights for target layers only
    base_weights = save_base_weights(model, target_layers)

    # Apply C3 edit in-place, measure, run, then restore
    apply_direction_ablation(
        model, directions, target_layers=target_layers, alpha=best_alpha, n_columns=64, inplace=True)

    # Measure edit size
    n_changed = sum(
        int((model.model.layers[li].mlp.down_proj.weight.data.cpu() - base_weights[li]).abs().gt(1e-6).sum())
        for li in target_layers
    )
    total_params = sum(p.numel() for p in model.parameters())
    edit_info = {"changed_params": n_changed, "total_params": total_params,
                 "fraction_changed": n_changed / total_params}
    print(f"\nC3 edit: {edit_info['changed_params']:,}/{edit_info['total_params']:,} params ({edit_info['fraction_changed']:.4%})")
    with open(RESULTS_DIR / "edit_info.json", "w") as f:
        json.dump(edit_info, f, indent=2)

    # -----------------------------------------------------------------------
    # Phase 3: TEST SET evaluation — one condition at a time, patch/restore
    # weights to keep GPU memory at 1 model at a time.
    # -----------------------------------------------------------------------
    print("\n[TEST] Running all conditions on TEST SET (touched once)...")
    from src.metrics import stratify_by_base_response, compute_perplexity
    import jsonlines as _jl
    summary = {}
    t0 = time.time()

    def run_en_regression(cname):
        """Run English test set for regression check."""
        test_en_dir = RESULTS_DIR / "test_en"
        test_en_dir.mkdir(exist_ok=True)
        recs = []
        for p in test_en:
            msgs = [{"role": "user", "content": p["prompt"]}]
            text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
            enc = {k: v.to(device) for k, v in enc.items()}
            inp_len = enc["input_ids"].shape[1]
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=60, do_sample=False,
                                     pad_token_id=tokenizer.eos_token_id)
            resp = tokenizer.decode(out[0][inp_len:], skip_special_tokens=True).strip()
            recs.append({"id": p["id"], "condition": cname + "_en", "response": resp,
                         "source_text": p["source_text"]})
        with _jl.open(str(test_en_dir / f"{cname}_en.jsonl"), "w") as w:
            w.write_all(recs)
        return eval_records(recs, "en")["lang_id_accuracy"]

    # C0: base model (no edit, no sys prompt)
    restore_weights(model, base_weights)   # ensure clean state
    c0 = run_condition(test_sv, "C0_baseline")
    summary["C0_baseline"] = eval_records(c0, "sv")
    summary["C0_baseline"]["english_lang_id"] = run_en_regression("C0_baseline")
    ppl0 = compute_perplexity(model, tokenizer, PPL_SENTENCES, device=device, batch_size=4)
    summary["C0_baseline"]["perplexity_sv"] = ppl0
    print(f"  C0: {summary['C0_baseline']['lang_id_accuracy']:.2%}  EN:{summary['C0_baseline']['english_lang_id']:.2%}  PPL:{ppl0:.1f}  ({time.time()-t0:.0f}s)")
    # checkpoint
    with open(RESULTS_DIR / "summary.json", "w") as f: json.dump(summary, f, indent=2)

    # C1: Swedish sys prompt
    restore_weights(model, base_weights)
    c1 = run_condition(test_sv, "C1_sysprompt", system_prompt="Svara alltid på svenska.")
    summary["C1_sysprompt"] = eval_records(c1, "sv")
    print(f"  C1: {summary['C1_sysprompt']['lang_id_accuracy']:.2%}")
    with open(RESULTS_DIR / "summary.json", "w") as f: json.dump(summary, f, indent=2)

    # C2: Stronger Swedish sys prompt
    restore_weights(model, base_weights)
    c2 = run_condition(test_sv, "C2_strong_sysprompt",
                       system_prompt="Du ar en svensk assistent. Svara alltid pa svenska och anvand korrekt svensk grammatik.")
    summary["C2_strong_sysprompt"] = eval_records(c2, "sv")
    print(f"  C2: {summary['C2_strong_sysprompt']['lang_id_accuracy']:.2%}")
    with open(RESULTS_DIR / "summary.json", "w") as f: json.dump(summary, f, indent=2)

    # C3: targeted direction ablation (apply in-place, run, then restore)
    restore_weights(model, base_weights)
    apply_direction_ablation(model, directions, target_layers=target_layers,
                             alpha=best_alpha, n_columns=64, inplace=True)
    c3 = run_condition(test_sv, "C3_direction_ablation")
    summary["C3_direction_ablation"] = eval_records(c3, "sv")
    # English regression with C3 model
    summary["C3_direction_ablation"]["english_lang_id"] = run_en_regression("C3_direction_ablation")
    ppl3 = compute_perplexity(model, tokenizer, PPL_SENTENCES, device=device, batch_size=4)
    summary["C3_direction_ablation"]["perplexity_sv"] = ppl3
    # Stratified analysis (failure removal vs new capability)
    strata = stratify_by_base_response(c0, c3, expected_lang="sv")
    summary["C3_direction_ablation"]["stratum_A"] = strata.get("A", {})
    summary["C3_direction_ablation"]["stratum_B"] = strata.get("B", {})
    print(f"  C3: {summary['C3_direction_ablation']['lang_id_accuracy']:.2%}  EN:{summary['C3_direction_ablation']['english_lang_id']:.2%}  PPL:{ppl3:.1f}")
    print(f"    Stratum A (n={strata.get('A',{}).get('n',0)}): edit={strata.get('A',{}).get('edit_accuracy',0):.2%}")
    print(f"    Stratum B (n={strata.get('B',{}).get('n',0)}): edit={strata.get('B',{}).get('edit_accuracy',0):.2%}")
    with open(RESULTS_DIR / "summary.json", "w") as f: json.dump(summary, f, indent=2)

    # C4: random direction controls (5 seeds, one at a time)
    c4_accs = []
    for si in range(5):
        restore_weights(model, base_weights)
        torch.manual_seed(si)
        rand_d = torch.randn(directions[top_layer].shape[0])
        rand_d = rand_d / (rand_d.norm() + 1e-8)
        rand_dirs = {top_layer: rand_d}
        apply_direction_ablation(model, rand_dirs, target_layers=target_layers,
                                  alpha=best_alpha, n_columns=64, inplace=True)
        recs = run_condition(test_sv, f"C4_random_s{si}")
        c4_accs.append(eval_records(recs, "sv")["lang_id_accuracy"])
        print(f"  C4 seed={si}: {c4_accs[-1]:.2%}")
    restore_weights(model, base_weights)
    summary["C4_random_control"] = {
        "lang_id_accuracy": float(np.mean(c4_accs)),
        "lang_id_std": float(np.std(c4_accs)), "n_seeds": 5}
    print(f"  C4 random: {np.mean(c4_accs):.2%} +/- {np.std(c4_accs):.2%}")
    with open(RESULTS_DIR / "summary.json", "w") as f: json.dump(summary, f, indent=2)

    # C5: sign-flipped edit
    restore_weights(model, base_weights)
    flipped_dirs = {top_layer: -directions[top_layer]}
    apply_direction_ablation(model, flipped_dirs, target_layers=target_layers,
                              alpha=best_alpha, n_columns=64, inplace=True)
    c5 = run_condition(test_sv, "C5_sign_flipped")
    summary["C5_sign_flipped"] = eval_records(c5, "sv")
    print(f"  C5 sign-flip: {summary['C5_sign_flipped']['lang_id_accuracy']:.2%}")
    restore_weights(model, base_weights)
    with open(RESULTS_DIR / "summary.json", "w") as f: json.dump(summary, f, indent=2)

    # C6: LoRA fine-tuning (uses PEFT, no deepcopy needed — PEFT adds adapter params)
    print("\n  Training LoRA (C6)...")
    try:
        from peft import get_peft_model, LoraConfig, TaskType
        restore_weights(model, base_weights)
        lora_cfg = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj","v_proj"],
                              lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM)
        peft_model = get_peft_model(model, lora_cfg)
        optimizer = torch.optim.AdamW(peft_model.parameters(), lr=2e-4)
        peft_model.train()
        for epoch in range(2):
            total_loss = 0.0
            for p in dev_sv:
                full_text = (tokenizer.apply_chat_template(
                    [{"role": "user", "content": p["prompt"]}],
                    tokenize=False, add_generation_prompt=True) + p["source_text"] + tokenizer.eos_token)
                enc = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=256)
                enc = {k: v.to(device) for k, v in enc.items()}
                in_len = len(tokenizer(tokenizer.apply_chat_template(
                    [{"role": "user", "content": p["prompt"]}],
                    tokenize=False, add_generation_prompt=True))["input_ids"])
                labels = enc["input_ids"].clone()
                labels[:, :in_len] = -100
                loss = peft_model(**enc, labels=labels).loss
                loss.backward(); optimizer.step(); optimizer.zero_grad()
                total_loss += loss.item()
            print(f"    LoRA epoch {epoch+1}/2: loss={total_loss/len(dev_sv):.4f}")
        peft_model.eval()
        c6 = run_condition(test_sv, "C6_lora", model_to_use=peft_model)
        summary["C6_lora"] = eval_records(c6, "sv")
        print(f"  C6 LoRA: {summary['C6_lora']['lang_id_accuracy']:.2%}")
        # Remove LoRA adapters to free memory and restore clean base
        peft_model.disable_adapter_layers()
        restore_weights(model, base_weights)
    except Exception as e:
        print(f"  C6 LoRA failed: {e}")
        restore_weights(model, base_weights)
    with open(RESULTS_DIR / "summary.json", "w") as f: json.dump(summary, f, indent=2)

    # Save summary
    with open(RESULTS_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nFull summary saved to {RESULTS_DIR / 'summary.json'}")

    # Save example outputs
    from src.metrics import detect_language
    import jsonlines
    ex_dir = RESULTS_DIR / "examples"
    ex_dir.mkdir(exist_ok=True)
    for cname, recs in [("C0_baseline", c0), ("C3_direction_ablation", c3)]:
        succ = [r for r in recs if detect_language(r.get("response","")) in ("sv","no","da")]
        fail = [r for r in recs if detect_language(r.get("response","")) == "en"]
        with jsonlines.open(str(ex_dir / f"{cname}_successes.jsonl"), "w") as w:
            w.write_all(succ[:30])
        with jsonlines.open(str(ex_dir / f"{cname}_failures.jsonl"), "w") as w:
            w.write_all(fail[:30])

    # Print final table
    print("\n" + "=" * 80)
    print(f"{'Condition':<30} {'LangID%':>8} {'CI_lo':>8} {'CI_hi':>8} {'PPL_sv':>8} {'EN%':>7}")
    print("-" * 80)
    for cond, m in summary.items():
        if "lang_id_accuracy" not in m:
            acc = m.get("lang_id_accuracy", 0)
            std = m.get("lang_id_std", 0)
            print(f"{cond:<30} {acc*100:>7.1f}% ±{std*100:.1f}%")
            continue
        pct = m["lang_id_accuracy"] * 100
        lo = m.get("lang_id_ci_lower", m["lang_id_accuracy"]) * 100
        hi = m.get("lang_id_ci_upper", m["lang_id_accuracy"]) * 100
        ppl = m.get("perplexity_sv", float("nan"))
        en = m.get("english_lang_id", float("nan"))
        en_s = f"{en*100:>6.1f}%" if en == en else "     -"
        print(f"{cond:<30} {pct:>7.1f}% {lo:>7.1f}% {hi:>7.1f}% {ppl:>8.1f} {en_s}")
    print("=" * 80)
    print(f"\nTotal runtime: {(time.time()-t0)/60:.1f} min")
    print("Done. Run scripts/analyze_results.py to generate figures.")


if __name__ == "__main__":
    main()
