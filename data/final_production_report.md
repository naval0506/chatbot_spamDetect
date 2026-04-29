# 📊 Training Report

## Overall Metrics (Test Set)

| Metric | Value |
|--------|-------|
| Accuracy | 0.9107 |
| Precision | 0.9142 |
| Recall | 0.9107 |
| F1 Score | 0.9110 |

## 🚀 Optimisations Appliquées

### Architecture du Pipeline

Le pipeline a été entièrement refondu pour maximiser les performances :

1. **FeatureUnion** : Combinaison de deux extracteurs TF-IDF complémentaires :
   - **Word n-grams (1-3)** : 30 000 features, capture les expressions multi-mots (`how do i`, `my card`).
   - **Character n-grams (2-5)** : 50 000 features, capture les motifs infra-mot et les fautes de frappe.
2. **LinearSVC** : Support Vector Machine linéaire, plus performant que LogisticRegression sur les données textuelles à haute dimensionnalité.
3. **CalibratedClassifierCV** : Wrapper pour fournir `predict_proba()` avec calibration Platt (nécessaire pour le chatbot).
4. **class_weight='balanced'** : Pondération automatique des classes sous-représentées.
5. **min_df=2, max_df=0.95** : Filtrage du bruit (mots trop rares ou trop fréquents).

### Résultat
Ces optimisations ont permis une amélioration significative de la précision par rapport au modèle de base (TF-IDF simple + LogisticRegression).


## Classification Report (Per Class)

```
                                                  precision    recall  f1-score   support

                           Refund_not_showing_up       0.95      0.93      0.94        40
                                activate_my_card       1.00      0.95      0.97        40
                                       age_limit       0.98      1.00      0.99        40
                         apple_pay_or_google_pay       1.00      1.00      1.00        40
                                     atm_support       0.95      0.97      0.96        40
                                automatic_top_up       0.97      0.95      0.96        40
         balance_not_updated_after_bank_transfer       0.70      0.78      0.74        40
balance_not_updated_after_cheque_or_cash_deposit       0.95      0.95      0.95        40
                         beneficiary_not_allowed       0.97      0.95      0.96        40
                                 cancel_transfer       0.97      0.95      0.96        40
                            card_about_to_expire       1.00      0.97      0.99        40
                                 card_acceptance       0.94      0.85      0.89        40
                                    card_arrival       0.85      0.88      0.86        40
                          card_delivery_estimate       0.90      0.88      0.89        40
                                    card_linking       0.95      0.97      0.96        40
                                card_not_working       0.70      0.95      0.81        40
                        card_payment_fee_charged       0.92      0.90      0.91        40
                     card_payment_not_recognised       0.89      0.85      0.87        40
                card_payment_wrong_exchange_rate       0.88      0.93      0.90        40
                                  card_swallowed       0.97      0.90      0.94        40
                          cash_withdrawal_charge       0.95      0.93      0.94        40
                  cash_withdrawal_not_recognised       0.88      0.95      0.92        40
                                      change_pin       0.97      0.95      0.96        40
                                compromised_card       0.90      0.93      0.91        40
                         contactless_not_working       1.00      0.93      0.96        40
                                 country_support       0.93      0.97      0.95        40
                           declined_card_payment       0.87      0.97      0.92        40
                        declined_cash_withdrawal       0.93      0.93      0.93        40
                               declined_transfer       0.97      0.82      0.89        40
             direct_debit_payment_not_recognised       0.88      0.88      0.88        40
                          disposable_card_limits       0.92      0.85      0.88        40
                           edit_personal_details       0.98      1.00      0.99        40
                                 exchange_charge       0.88      0.90      0.89        40
                                   exchange_rate       0.93      0.95      0.94        40
                                exchange_via_app       0.88      0.90      0.89        40
                       extra_charge_on_statement       0.95      0.88      0.91        40
                                 failed_transfer       0.74      0.88      0.80        40
                           fiat_currency_support       0.97      0.80      0.88        40
                     get_disposable_virtual_card       0.92      0.82      0.87        40
                               get_physical_card       0.89      0.97      0.93        40
                              getting_spare_card       0.93      0.93      0.93        40
                            getting_virtual_card       0.79      0.95      0.86        40
                             lost_or_stolen_card       0.84      0.95      0.89        40
                            lost_or_stolen_phone       1.00      0.97      0.99        40
                             order_physical_card       0.88      0.88      0.88        40
                              passcode_forgotten       1.00      1.00      1.00        40
                            pending_card_payment       0.90      0.90      0.90        40
                         pending_cash_withdrawal       0.95      0.90      0.92        40
                                  pending_top_up       0.90      0.90      0.90        40
                                pending_transfer       0.81      0.75      0.78        40
                                     pin_blocked       0.94      0.80      0.86        40
                                 receiving_money       0.97      0.95      0.96        40
                                  request_refund       0.95      0.93      0.94        40
                          reverted_card_payment?       0.93      0.95      0.94        40
                  supported_cards_and_currencies       0.85      1.00      0.92        40
                               terminate_account       1.00      1.00      1.00        40
                  top_up_by_bank_transfer_charge       0.95      0.88      0.91        40
                           top_up_by_card_charge       0.93      0.97      0.95        40
                        top_up_by_cash_or_cheque       0.92      0.88      0.90        40
                                   top_up_failed       0.84      0.78      0.81        40
                                   top_up_limits       0.88      0.95      0.92        40
                                 top_up_reverted       0.82      0.82      0.82        40
                              topping_up_by_card       0.95      0.88      0.91        40
                       transaction_charged_twice       0.91      1.00      0.95        40
                            transfer_fee_charged       0.92      0.88      0.90        40
                           transfer_into_account       0.86      0.90      0.88        40
              transfer_not_received_by_recipient       0.83      0.85      0.84        40
                                 transfer_timing       0.94      0.82      0.88        40
                       unable_to_verify_identity       0.89      0.82      0.86        40
                              verify_my_identity       0.82      0.82      0.82        40
                          verify_source_of_funds       0.91      1.00      0.95        40
                                   verify_top_up       1.00      1.00      1.00        40
                        virtual_card_not_working       1.00      0.85      0.92        40
                              visa_or_mastercard       1.00      0.93      0.96        40
                             why_verify_identity       0.81      0.88      0.84        40
                   wrong_amount_of_cash_received       0.92      0.90      0.91        40
         wrong_exchange_rate_for_cash_withdrawal       0.85      0.88      0.86        40

                                        accuracy                           0.91      3080
                                       macro avg       0.91      0.91      0.91      3080
                                    weighted avg       0.91      0.91      0.91      3080

```