"""
Jour 1 - Données : Chargement et nettoyage du dataset banking77
"""

import pandas as pd, re, unicodedata, os, sys

DATA_DIR  = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TEST_CSV  = os.path.join(DATA_DIR, "test.csv")

# Réponses prédéfinies pour chaque catégorie banking77
REPONSES = {
    "activate_my_card":              "To activate your card, go to the app under 'My Cards' and tap 'Activate'. You can also call our support line.",
    "age_limit":                     "You must be at least 18 years old to open an account.",
    "apple_pay_or_google_pay":       "Add your card to Apple Pay or Google Pay via the app under 'Cards > Add to Wallet'.",
    "atm_support":                   "You can withdraw cash at any ATM worldwide. Fees may apply depending on your plan.",
    "automatic_top_up":              "Enable automatic top-up in the app: Top Up > Automatic. Set your threshold and amount.",
    "balance_not_updated":           "Balances update in real time. Pull to refresh. Contact support if the issue persists.",
    "beneficiary_not_allowed":       "Some beneficiaries are restricted for security. Contact support to review your case.",
    "cancel_transfer":               "Instant transfers cannot be cancelled. Scheduled transfers can be cancelled before execution date.",
    "card_about_to_expire":          "A new card is sent automatically before expiry. Check the expected delivery date in the app.",
    "card_acceptance":               "Your card is accepted at all Visa/Mastercard merchants worldwide.",
    "card_arrival":                  "New cards arrive within 3–5 business days. Track delivery in the app under 'My Cards'.",
    "card_delivery_estimate":        "Standard delivery takes 3–5 business days. Express options may be available in the app.",
    "card_linked_accounts":          "Link multiple accounts to your card in the app under 'Cards > Linked Accounts'.",
    "card_not_working":              "Ensure your card is activated and has sufficient funds. Try another terminal or contact support.",
    "card_payment_fee":              "Card payments in your home currency are free. Foreign currency payments may have a small fee.",
    "card_payment_not_recognised":   "Unrecognised payments may take 24h to appear. If still missing after 7 days, contact support.",
    "card_payment_wrong_amount":     "If charged the wrong amount, contact the merchant first. If unresolved, open a dispute in the app.",
    "card_swallowed":                "If your card was swallowed by an ATM, block it immediately in the app and order a replacement.",
    "cash_withdrawal_charge":        "ATM withdrawal fees depend on your plan. Check your fee schedule in the app.",
    "cash_withdrawal_not_recognised":"If you don't recognise a cash withdrawal, block your card immediately and contact support.",
    "change_pin":                    "Change your PIN in the app under 'Cards > Change PIN', or at any compatible ATM.",
    "compromised_card":              "If your card is compromised, block it immediately in the app and order a replacement.",
    "contactless_not_working":       "Enable contactless in the app under 'Cards > Contactless'. Ensure NFC is on on your phone.",
    "country_support":               "We operate in over 30 countries. Check the full list on our website.",
    "declined_card_payment":         "Declines can be due to insufficient funds, spending limits, or merchant restrictions. Check the app.",
    "declined_cash_withdrawal":      "Withdrawals can be declined due to daily limits or ATM restrictions. Check your limits in the app.",
    "declined_transfer":             "Transfers can be declined due to security checks or account limits. Contact support for details.",
    "direct_debit_payment_not_recognised": "Check the merchant name on your statement. Contact support if you still don't recognise it.",
    "disposable_card_virtual_card":  "Create a disposable virtual card in the app under 'Cards > New Virtual Card'.",
    "edit_personal_details":         "Update your personal details in the app under 'Profile > Edit'. Some changes may require ID verification.",
    "exchange_charge":               "We use the real exchange rate with a small markup. Check the exact rate before any foreign transaction.",
    "exchange_rate":                 "We offer competitive exchange rates close to the interbank rate. Check current rates in the app.",
    "exchange_via_app":              "Exchange currencies instantly in the app under 'Exchange'. Rates are shown before you confirm.",
    "extra_charge_on_statement":     "Extra charges may be pre-authorisation holds. They usually drop off within 7 days.",
    "failed_transfer":               "Failed transfers are refunded immediately. Common causes: incorrect IBAN, limits, or security checks.",
    "fiat_currency_support":         "We support all major fiat currencies. Check the full list in the app under 'Exchange'.",
    "freeze_account":                "Freeze your account instantly in the app under 'Settings > Security > Freeze Account'.",
    "get_disposable_virtual_card":   "Get a disposable virtual card in the app under 'Cards > Virtual > Disposable'.",
    "get_physical_card":             "Order a physical card in the app under 'Cards > Order Physical Card'. Delivery in 3–5 business days.",
    "getting_spare_card":            "Order an extra card in the app under 'Cards > Additional Cards'.",
    "getting_virtual_card":          "Get a virtual card instantly in the app under 'Cards > New Card > Virtual'.",
    "lost_or_stolen_card":           "Block your card immediately in the app under 'Cards > Block', then order a replacement.",
    "lost_or_stolen_phone":          "If your phone is lost, freeze your account from any browser. Change your password immediately.",
    "order_physical_card":           "Order a physical card in the app under 'Cards > Order Physical Card'. Free with most plans.",
    "passcode_forgotten":            "Reset your passcode using biometrics or your registered email on the login screen.",
    "pending_card_payment":          "Pending payments are pre-authorisations that settle within 7 days.",
    "pending_cash_withdrawal":       "Pending withdrawals settle within 24 hours. Contact support if still pending after that.",
    "pending_top_up":                "Top-ups usually appear instantly. If pending for more than 10 minutes, contact support.",
    "pending_transfer":              "Transfers are usually instant. If pending, check the recipient's details and your transfer limits.",
    "pin_blocked":                   "Your PIN is blocked after 3 wrong attempts. Unblock it in the app under 'Cards > Unblock PIN'.",
    "receiving_money":               "Share your IBAN or account number (found in app under 'Account Details') to receive money.",
    "Refund_not_showing":            "Refunds take 3–5 business days. If not showing after 7 days, contact the merchant.",
    "request_refund":                "Request a refund from the merchant first. If refused, open a dispute in the app under 'Transactions > Dispute'.",
    "reverted_card_payment":         "Reverted payments return funds to your account within 3 business days.",
    "supported_cards_and_currencies":"We support Visa and Mastercard in 150+ currencies.",
    "terminate_account":             "To close your account, go to 'Profile > Close Account'. Ensure your balance is zero first.",
    "top_up_by_bank_transfer":       "Top up by bank transfer using the account details in the app. Arrives within 1 business day.",
    "top_up_by_card":                "Top up instantly with any debit or credit card in the app under 'Top Up > Card'.",
    "top_up_by_cash_or_cheque":      "Cash and cheque top-ups are not supported. Use bank transfer or card top-up instead.",
    "top_up_failed":                 "Failed top-ups are not charged. Common causes: card declined, wrong details, or daily limits.",
    "top_up_limits":                 "Top-up limits depend on your plan and verification level. Check your current limits in the app.",
    "top_up_reverted":               "Reverted top-ups return funds to your source. Check with your bank and try again.",
    "topping_up_by_card":            "Top up instantly with any Visa or Mastercard debit card in the app under 'Top Up'.",
    "transaction_charged_twice":     "Double charges may be a pre-auth and settlement. If both settle, open a dispute in the app.",
    "transfer_fee_charged":          "Transfer fees depend on the destination and your plan. Review fees before sending.",
    "transfer_into_account":         "To receive a transfer, share your IBAN found in the app under 'Account Details'.",
    "transfer_limit_reached":        "You've reached your transfer limit. Increase it temporarily in the app or contact support.",
    "transfer_not_received_by_recipient": "Check the IBAN is correct. Transfers can take up to 2 business days. Contact support if delayed.",
    "transfer_timing":               "SEPA transfers arrive the next business day. Instant transfers arrive in seconds.",
    "unable_to_verify_identity":     "Identity verification needs a valid passport or ID card. Ensure the photo is clear and not expired.",
    "unknown_top_up":                "Check if a family member made the top-up. Contact support if the source is still unclear.",
    "verify_my_identity":            "Verify your identity in the app under 'Profile > Verify Identity'. You'll need a valid ID and a selfie.",
    "verify_source_of_funds":        "Source of funds verification may be required for large transactions. Upload documents in the app.",
    "verify_top_up":                 "Some top-ups require verification. Follow the in-app instructions to confirm the source.",
    "virtual_card_not_working":      "Ensure your virtual card is active in the app. Check that the merchant accepts virtual cards.",
    "visa_or_mastercard":            "We issue both Visa and Mastercard. The type depends on your account type and location.",
    "why_verify_identity":           "Identity verification is required by law (KYC/AML) to protect you and prevent fraud.",
    "wrong_amount_of_cash_received": "If an ATM gave the wrong amount, report it to the operator and contact our support.",
    "wrong_exchange_rate_for_cash_withdrawal": "Exchange rates for ATM withdrawals are set by the ATM operator, not by us.",
}


def verifier_dataset():
    manquants = [f for f in [TRAIN_CSV, TEST_CSV] if not os.path.exists(f)]
    if manquants:
        print("=" * 55)
        print("DATASET MANQUANT !")
        print("Lance d'abord :  python setup.py")
        print("=" * 55)
        sys.exit(1)


def normaliser_texte(texte: str) -> str:
    if not isinstance(texte, str):
        return ""
    texte = texte.lower()
    texte = unicodedata.normalize("NFD", texte)
    texte = "".join(c for c in texte if unicodedata.category(c) != "Mn")
    texte = re.sub(r"[^\w\s']", " ", texte)
    return re.sub(r"\s+", " ", texte).strip()


def charger_banking77() -> pd.DataFrame:
    verifier_dataset()
    df = pd.concat([pd.read_csv(TRAIN_CSV), pd.read_csv(TEST_CSV)], ignore_index=True)
    df.columns = [c.strip().lower() for c in df.columns]
    if "category" in df.columns:
        df.rename(columns={"category": "label"}, inplace=True)
    print(f"[INFO] Dataset : {len(df)} entrées, {df['label'].nunique()} catégories.")
    return df


def construire_faq(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, grp in df.groupby("label"):
        reponse = REPONSES.get(label, f"For questions about '{label.replace('_', ' ')}', please contact our support team.")
        for _, row in grp.iterrows():
            rows.append({
                "question":      row["text"],
                "reponse":       reponse,
                "categorie":     label,
                "question_clean": normaliser_texte(row["text"])
            })
    faq = pd.DataFrame(rows).drop_duplicates(subset=["question_clean"]).reset_index(drop=True)
    print(f"[INFO] FAQ : {len(faq)} entrées, {faq['categorie'].nunique()} catégories.")
    return faq
