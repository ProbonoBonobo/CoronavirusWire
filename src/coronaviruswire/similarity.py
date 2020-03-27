from itertools import combinations_with_replacement
import spacy
print(f"Loading vector model...")
nlp = spacy.load("en_core_web_lg")
print(f"Model loaded.")

headlines = (
    "Banks agree to 90-day grace period on mortgage payments for California "
    "families impacted by coronavirus, Gov. Newsom says",
    "Coronavirus: 4 major banks agree to 90-day grace period for mortgage "
    "payments in CA, Newsom says",
    "Coronavirus: Banks agree to temporarily waive mortgage fees in CA",
    "Lancaster teen's death no longer counted among LA County's coronavirus total",
    "Gas prices drop under $2 at Jurupa Valley truck stop amid coronavirus pandemic",
    "Coronavirus Southern California update: 3 additional deaths, 138 new cases "
    "confirmed in Los Angeles County",
    "13 people have died of COVID-19 in L.A. County; officials now excluding Lancaster "
    "teen’s death pending CDC investigation",
    "Simply Salad has a clever way to get you to order in with them",
    "'Top Chef' winner Floyd Cardoz dies due to coronavirus complications",
    "O.C. reports 187 cases of the coronavirus, a 50% jump in 2 days",
    "Nearly half of all patients at Kaiser hospital in San Jose believed to have coronavirus"
)
docs = [nlp(headline) for headline in headlines]
k = len(docs)
for i, doc1 in enumerate(docs):
    print(f"Calculating {k} nearest neighbors for Headline #{i}: '{doc1}'")
    simvec = [(doc2.similarity(doc1), doc2) for doc2 in docs]
    gradient = sorted(simvec, reverse=True)
    for j, neighbor in enumerate(gradient):
        similarity, headline = neighbor
        print(f"    {j+1}. ({round(similarity, 3)*100}% similar) {headline}")
    print(
        "\n\n============================================================================\n\n"
    )
"""
Output:
/home/kz/.local/share/virtualenvs/coronaviruswire-v4DtK_G7/bin/python /home/kz/projects/coronaviruswire/src/coronaviruswire/similarity.py
Loading vector model...
Model loaded.
Calculating 11 nearest neighbors for Headline #0: 'Banks agree to 90-day grace period on mortgage payments for California families impacted by coronavirus, Gov. Newsom says'
    1. (100.0% similar) Banks agree to 90-day grace period on mortgage payments for California families impacted by coronavirus, Gov. Newsom says
    2. (92.0% similar) Coronavirus: 4 major banks agree to 90-day grace period for mortgage payments in CA, Newsom says
    3. (83.5% similar) Coronavirus: Banks agree to temporarily waive mortgage fees in CA
    4. (77.8% similar) 13 people have died of COVID-19 in L.A. County; officials now excluding Lancaster teen’s death pending CDC investigation
    5. (75.8% similar) Nearly half of all patients at Kaiser hospital in San Jose believed to have coronavirus
    6. (73.7% similar) Lancaster teen's death no longer counted among LA County's coronavirus total
    7. (69.69999999999999% similar) Coronavirus Southern California update: 3 additional deaths, 138 new cases confirmed in Los Angeles County
    8. (68.7% similar) Gas prices drop under $2 at Jurupa Valley truck stop amid coronavirus pandemic
    9. (67.9% similar) Simply Salad has a clever way to get you to order in with them
    10. (66.10000000000001% similar) O.C. reports 187 cases of the coronavirus, a 50% jump in 2 days
    11. (57.99999999999999% similar) 'Top Chef' winner Floyd Cardoz dies due to coronavirus complications


============================================================================


Calculating 11 nearest neighbors for Headline #1: 'Coronavirus: 4 major banks agree to 90-day grace period for mortgage payments in CA, Newsom says'
    1. (100.0% similar) Coronavirus: 4 major banks agree to 90-day grace period for mortgage payments in CA, Newsom says
    2. (92.0% similar) Banks agree to 90-day grace period on mortgage payments for California families impacted by coronavirus, Gov. Newsom says
    3. (88.8% similar) Coronavirus: Banks agree to temporarily waive mortgage fees in CA
    4. (76.9% similar) 13 people have died of COVID-19 in L.A. County; officials now excluding Lancaster teen’s death pending CDC investigation
    5. (75.4% similar) Simply Salad has a clever way to get you to order in with them
    6. (73.1% similar) Nearly half of all patients at Kaiser hospital in San Jose believed to have coronavirus
    7. (72.39999999999999% similar) Coronavirus Southern California update: 3 additional deaths, 138 new cases confirmed in Los Angeles County
    8. (72.0% similar) Lancaster teen's death no longer counted among LA County's coronavirus total
    9. (70.8% similar) O.C. reports 187 cases of the coronavirus, a 50% jump in 2 days
    10. (66.9% similar) Gas prices drop under $2 at Jurupa Valley truck stop amid coronavirus pandemic
    11. (57.3% similar) 'Top Chef' winner Floyd Cardoz dies due to coronavirus complications


============================================================================


Calculating 11 nearest neighbors for Headline #2: 'Coronavirus: Banks agree to temporarily waive mortgage fees in CA'
    1. (100.0% similar) Coronavirus: Banks agree to temporarily waive mortgage fees in CA
    2. (88.8% similar) Coronavirus: 4 major banks agree to 90-day grace period for mortgage payments in CA, Newsom says
    3. (83.5% similar) Banks agree to 90-day grace period on mortgage payments for California families impacted by coronavirus, Gov. Newsom says
    4. (72.2% similar) Simply Salad has a clever way to get you to order in with them
    5. (69.19999999999999% similar) 13 people have died of COVID-19 in L.A. County; officials now excluding Lancaster teen’s death pending CDC investigation
    6. (66.8% similar) Nearly half of all patients at Kaiser hospital in San Jose believed to have coronavirus
    7. (64.7% similar) Coronavirus Southern California update: 3 additional deaths, 138 new cases confirmed in Los Angeles County
    8. (63.3% similar) Gas prices drop under $2 at Jurupa Valley truck stop amid coronavirus pandemic
    9. (63.2% similar) Lancaster teen's death no longer counted among LA County's coronavirus total
    10. (61.0% similar) O.C. reports 187 cases of the coronavirus, a 50% jump in 2 days
    11. (52.0% similar) 'Top Chef' winner Floyd Cardoz dies due to coronavirus complications


============================================================================


Calculating 11 nearest neighbors for Headline #3: 'Lancaster teen's death no longer counted among LA County's coronavirus total'
    1. (100.0% similar) Lancaster teen's death no longer counted among LA County's coronavirus total
    2. (87.4% similar) 13 people have died of COVID-19 in L.A. County; officials now excluding Lancaster teen’s death pending CDC investigation
    3. (82.6% similar) Nearly half of all patients at Kaiser hospital in San Jose believed to have coronavirus
    4. (78.2% similar) Coronavirus Southern California update: 3 additional deaths, 138 new cases confirmed in Los Angeles County
    5. (74.2% similar) O.C. reports 187 cases of the coronavirus, a 50% jump in 2 days
    6. (73.7% similar) Banks agree to 90-day grace period on mortgage payments for California families impacted by coronavirus, Gov. Newsom says
    7. (72.0% similar) Coronavirus: 4 major banks agree to 90-day grace period for mortgage payments in CA, Newsom says
    8. (71.5% similar) 'Top Chef' winner Floyd Cardoz dies due to coronavirus complications
    9. (68.4% similar) Gas prices drop under $2 at Jurupa Valley truck stop amid coronavirus pandemic
    10. (67.30000000000001% similar) Simply Salad has a clever way to get you to order in with them
    11. (63.2% similar) Coronavirus: Banks agree to temporarily waive mortgage fees in CA


============================================================================


Calculating 11 nearest neighbors for Headline #4: 'Gas prices drop under $2 at Jurupa Valley truck stop amid coronavirus pandemic'
    1. (100.0% similar) Gas prices drop under $2 at Jurupa Valley truck stop amid coronavirus pandemic
    2. (73.9% similar) Nearly half of all patients at Kaiser hospital in San Jose believed to have coronavirus
    3. (73.8% similar) O.C. reports 187 cases of the coronavirus, a 50% jump in 2 days
    4. (71.8% similar) 13 people have died of COVID-19 in L.A. County; officials now excluding Lancaster teen’s death pending CDC investigation
    5. (69.69999999999999% similar) Coronavirus Southern California update: 3 additional deaths, 138 new cases confirmed in Los Angeles County
    6. (68.7% similar) Banks agree to 90-day grace period on mortgage payments for California families impacted by coronavirus, Gov. Newsom says
    7. (68.4% similar) Lancaster teen's death no longer counted among LA County's coronavirus total
    8. (66.9% similar) Coronavirus: 4 major banks agree to 90-day grace period for mortgage payments in CA, Newsom says
    9. (65.5% similar) Simply Salad has a clever way to get you to order in with them
    10. (63.3% similar) Coronavirus: Banks agree to temporarily waive mortgage fees in CA
    11. (61.7% similar) 'Top Chef' winner Floyd Cardoz dies due to coronavirus complications


============================================================================


Calculating 11 nearest neighbors for Headline #5: 'Coronavirus Southern California update: 3 additional deaths, 138 new cases confirmed in Los Angeles County'
    1. (100.0% similar) Coronavirus Southern California update: 3 additional deaths, 138 new cases confirmed in Los Angeles County
    2. (85.5% similar) 13 people have died of COVID-19 in L.A. County; officials now excluding Lancaster teen’s death pending CDC investigation
    3. (78.4% similar) Nearly half of all patients at Kaiser hospital in San Jose believed to have coronavirus
    4. (78.2% similar) Lancaster teen's death no longer counted among LA County's coronavirus total
    5. (75.2% similar) O.C. reports 187 cases of the coronavirus, a 50% jump in 2 days
    6. (72.39999999999999% similar) Coronavirus: 4 major banks agree to 90-day grace period for mortgage payments in CA, Newsom says
    7. (69.69999999999999% similar) Gas prices drop under $2 at Jurupa Valley truck stop amid coronavirus pandemic
    8. (69.69999999999999% similar) Banks agree to 90-day grace period on mortgage payments for California families impacted by coronavirus, Gov. Newsom says
    9. (64.7% similar) Coronavirus: Banks agree to temporarily waive mortgage fees in CA
    10. (59.699999999999996% similar) Simply Salad has a clever way to get you to order in with them
    11. (58.599999999999994% similar) 'Top Chef' winner Floyd Cardoz dies due to coronavirus complications


============================================================================


Calculating 11 nearest neighbors for Headline #6: '13 people have died of COVID-19 in L.A. County; officials now excluding Lancaster teen’s death pending CDC investigation'
    1. (100.0% similar) 13 people have died of COVID-19 in L.A. County; officials now excluding Lancaster teen’s death pending CDC investigation
    2. (87.4% similar) Lancaster teen's death no longer counted among LA County's coronavirus total
    3. (85.5% similar) Coronavirus Southern California update: 3 additional deaths, 138 new cases confirmed in Los Angeles County
    4. (83.2% similar) Nearly half of all patients at Kaiser hospital in San Jose believed to have coronavirus
    5. (77.8% similar) Banks agree to 90-day grace period on mortgage payments for California families impacted by coronavirus, Gov. Newsom says
    6. (77.0% similar) O.C. reports 187 cases of the coronavirus, a 50% jump in 2 days
    7. (76.9% similar) Coronavirus: 4 major banks agree to 90-day grace period for mortgage payments in CA, Newsom says
    8. (71.8% similar) Gas prices drop under $2 at Jurupa Valley truck stop amid coronavirus pandemic
    9. (69.19999999999999% similar) Coronavirus: Banks agree to temporarily waive mortgage fees in CA
    10. (68.10000000000001% similar) Simply Salad has a clever way to get you to order in with them
    11. (66.4% similar) 'Top Chef' winner Floyd Cardoz dies due to coronavirus complications


============================================================================


Calculating 11 nearest neighbors for Headline #7: 'Simply Salad has a clever way to get you to order in with them'
    1. (100.0% similar) Simply Salad has a clever way to get you to order in with them
    2. (75.4% similar) Coronavirus: 4 major banks agree to 90-day grace period for mortgage payments in CA, Newsom says
    3. (74.2% similar) Nearly half of all patients at Kaiser hospital in San Jose believed to have coronavirus
    4. (72.2% similar) Coronavirus: Banks agree to temporarily waive mortgage fees in CA
    5. (69.19999999999999% similar) O.C. reports 187 cases of the coronavirus, a 50% jump in 2 days
    6. (68.10000000000001% similar) 13 people have died of COVID-19 in L.A. County; officials now excluding Lancaster teen’s death pending CDC investigation
    7. (67.9% similar) Banks agree to 90-day grace period on mortgage payments for California families impacted by coronavirus, Gov. Newsom says
    8. (67.30000000000001% similar) Lancaster teen's death no longer counted among LA County's coronavirus total
    9. (65.5% similar) Gas prices drop under $2 at Jurupa Valley truck stop amid coronavirus pandemic
    10. (63.800000000000004% similar) 'Top Chef' winner Floyd Cardoz dies due to coronavirus complications
    11. (59.699999999999996% similar) Coronavirus Southern California update: 3 additional deaths, 138 new cases confirmed in Los Angeles County


============================================================================


Calculating 11 nearest neighbors for Headline #8: ''Top Chef' winner Floyd Cardoz dies due to coronavirus complications'
    1. (100.0% similar) 'Top Chef' winner Floyd Cardoz dies due to coronavirus complications
    2. (71.5% similar) Lancaster teen's death no longer counted among LA County's coronavirus total
    3. (69.1% similar) Nearly half of all patients at Kaiser hospital in San Jose believed to have coronavirus
    4. (66.4% similar) 13 people have died of COVID-19 in L.A. County; officials now excluding Lancaster teen’s death pending CDC investigation
    5. (63.800000000000004% similar) Simply Salad has a clever way to get you to order in with them
    6. (63.6% similar) O.C. reports 187 cases of the coronavirus, a 50% jump in 2 days
    7. (61.7% similar) Gas prices drop under $2 at Jurupa Valley truck stop amid coronavirus pandemic
    8. (58.599999999999994% similar) Coronavirus Southern California update: 3 additional deaths, 138 new cases confirmed in Los Angeles County
    9. (57.99999999999999% similar) Banks agree to 90-day grace period on mortgage payments for California families impacted by coronavirus, Gov. Newsom says
    10. (57.3% similar) Coronavirus: 4 major banks agree to 90-day grace period for mortgage payments in CA, Newsom says
    11. (52.0% similar) Coronavirus: Banks agree to temporarily waive mortgage fees in CA


============================================================================


Calculating 11 nearest neighbors for Headline #9: 'O.C. reports 187 cases of the coronavirus, a 50% jump in 2 days'
    1. (100.0% similar) O.C. reports 187 cases of the coronavirus, a 50% jump in 2 days
    2. (77.5% similar) Nearly half of all patients at Kaiser hospital in San Jose believed to have coronavirus
    3. (77.0% similar) 13 people have died of COVID-19 in L.A. County; officials now excluding Lancaster teen’s death pending CDC investigation
    4. (75.2% similar) Coronavirus Southern California update: 3 additional deaths, 138 new cases confirmed in Los Angeles County
    5. (74.2% similar) Lancaster teen's death no longer counted among LA County's coronavirus total
    6. (73.8% similar) Gas prices drop under $2 at Jurupa Valley truck stop amid coronavirus pandemic
    7. (70.8% similar) Coronavirus: 4 major banks agree to 90-day grace period for mortgage payments in CA, Newsom says
    8. (69.19999999999999% similar) Simply Salad has a clever way to get you to order in with them
    9. (66.10000000000001% similar) Banks agree to 90-day grace period on mortgage payments for California families impacted by coronavirus, Gov. Newsom says
    10. (63.6% similar) 'Top Chef' winner Floyd Cardoz dies due to coronavirus complications
    11. (61.0% similar) Coronavirus: Banks agree to temporarily waive mortgage fees in CA


============================================================================


Calculating 11 nearest neighbors for Headline #10: 'Nearly half of all patients at Kaiser hospital in San Jose believed to have coronavirus'
    1. (100.0% similar) Nearly half of all patients at Kaiser hospital in San Jose believed to have coronavirus
    2. (83.2% similar) 13 people have died of COVID-19 in L.A. County; officials now excluding Lancaster teen’s death pending CDC investigation
    3. (82.6% similar) Lancaster teen's death no longer counted among LA County's coronavirus total
    4. (78.4% similar) Coronavirus Southern California update: 3 additional deaths, 138 new cases confirmed in Los Angeles County
    5. (77.5% similar) O.C. reports 187 cases of the coronavirus, a 50% jump in 2 days
    6. (75.8% similar) Banks agree to 90-day grace period on mortgage payments for California families impacted by coronavirus, Gov. Newsom says
    7. (74.2% similar) Simply Salad has a clever way to get you to order in with them
    8. (73.9% similar) Gas prices drop under $2 at Jurupa Valley truck stop amid coronavirus pandemic
    9. (73.1% similar) Coronavirus: 4 major banks agree to 90-day grace period for mortgage payments in CA, Newsom says
    10. (69.1% similar) 'Top Chef' winner Floyd Cardoz dies due to coronavirus complications
    11. (66.8% similar) Coronavirus: Banks agree to temporarily waive mortgage fees in CA


============================================================================
"""
