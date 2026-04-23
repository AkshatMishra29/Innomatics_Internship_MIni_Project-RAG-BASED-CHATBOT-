# ============================================================
# hitl.py — Human in the Loop Escalation Manager
# Decides when to escalate + manages support tickets
# ============================================================

import uuid
import logging
from datetime import datetime

# ─── LOGGING SETUP ───────────────────────────────────────────
logging.basicConfig(
    filename="escalation_log.txt",
    level=logging.INFO,
    format="%(asctime)s — %(message)s"
)

# ─── ESCALATION KEYWORDS ─────────────────────────────────────
ESCALATION_KEYWORDS = [
    "urgent", "complaint", "refund", "legal", "lawsuit",
    "fraud", "angry", "unacceptable", "scam", "cheat",
    "police", "court", "consumer forum", "harassment"
]


class HITLManager:
    """
    Human-in-the-Loop Manager.
    1. Checks if query needs human escalation
    2. Creates support ticket when escalated
    3. Stores and returns human agent responses
    """

    def __init__(self):
        # Stores all tickets in memory
        # Format: { ticket_id: { query, timestamp, status, response } }
        self.escalation_log = {}


    def check_escalation(self, query: str, chunks: list) -> bool:
        """
        Returns True if query needs human escalation.

        Escalates when ANY condition is true:
        1. Less than 2 chunks found (not enough context)
        2. Query contains sensitive keywords
        3. Query is longer than 200 characters
        """
        print("\n🔎 Checking escalation conditions ...")

        # Condition 1: Not enough context
        if len(chunks) < 2:
            print(f"   ⚠️  Only {len(chunks)} chunk(s) found. Escalating.")
            return True

        # Condition 2: Sensitive keywords found
        query_lower = query.lower()
        found = [kw for kw in ESCALATION_KEYWORDS if kw in query_lower]
        if found:
            print(f"   ⚠️  Keywords found: {found}. Escalating.")
            return True

        # Condition 3: Very long query
        if len(query) > 200:
            print(f"   ⚠️  Query too long ({len(query)} chars). Escalating.")
            return True

        print("   ✅ No escalation needed.")
        return False


    def handle_escalation(self, query: str) -> str:
        """
        Creates a support ticket and returns message to user.
        """
        print("\n🚨 Creating escalation ticket ...")

        # Generate unique ticket ID
        ticket_id = "TKT-" + str(uuid.uuid4())[:8].upper()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Store ticket
        self.escalation_log[ticket_id] = {
            "query"     : query,
            "timestamp" : timestamp,
            "status"    : "OPEN",
            "response"  : None
        }

        # Write to log file
        logging.info(f"TICKET {ticket_id} | {query[:100]} | {timestamp}")
        print(f"   ✅ Ticket created: {ticket_id}")

        message = (
            f"Your query has been escalated to a human agent.\n\n"
            f"🎫 Ticket ID  : {ticket_id}\n"
            f"🕐 Created At : {timestamp}\n"
            f"📋 Status     : OPEN\n\n"
            f"Our team will respond within 2-4 business hours.\n"
            f"Business hours: Monday to Saturday, 9AM to 8PM IST.\n\n"
            f"For urgent help call: 1800-123-4567 (Toll Free)"
        )

        return message


    def integrate_human_response(self, ticket_id: str,
                                  human_response: str) -> str:
        """
        Saves human agent response and returns it to user.
        """
        print(f"\n👨‍💼 Resolving ticket: {ticket_id}")

        if ticket_id not in self.escalation_log:
            return f"❌ Ticket '{ticket_id}' not found."

        # Update ticket
        self.escalation_log[ticket_id]["response"] = human_response
        self.escalation_log[ticket_id]["status"]   = "RESOLVED"
        resolved_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        print(f"   ✅ Ticket {ticket_id} resolved.")

        response = (
            f"✅ Your ticket has been resolved.\n\n"
            f"🎫 Ticket ID  : {ticket_id}\n"
            f"🕐 Resolved   : {resolved_at}\n"
            f"📋 Status     : RESOLVED\n\n"
            f"Agent Response:\n{human_response}"
        )

        return response


    def get_ticket_status(self, ticket_id: str) -> str:
        """
        Returns current status of a ticket.
        """
        if ticket_id not in self.escalation_log:
            return f"❌ Ticket '{ticket_id}' not found."

        t = self.escalation_log[ticket_id]
        return (
            f"🎫 Ticket  : {ticket_id}\n"
            f"📋 Status  : {t['status']}\n"
            f"🕐 Created : {t['timestamp']}\n"
            f"❓ Query   : {t['query'][:100]}"
        )


# ─── RUN ─────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("   HITL MODULE TEST")
    print("=" * 50)

    manager = HITLManager()

    # Test 1: Should NOT escalate
    q1 = "What is the price of SmartHome Hub?"
    c1 = ["SmartHome Hub costs Rs.8999", "Available in black and white"]
    print(f"\nTest 1 — Escalate? {manager.check_escalation(q1, c1)}")
    print("Expected: False")

    # Test 2: SHOULD escalate (keyword)
    q2 = "I want a refund, this is fraud!"
    c2 = ["Return policy is 10 days", "Refund in 5-7 days"]
    print(f"\nTest 2 — Escalate? {manager.check_escalation(q2, c2)}")
    print("Expected: True")

    # Test 3: Create ticket
    print("\nTest 3 — Creating ticket:")
    msg = manager.handle_escalation(q2)
    print(msg)

    # Test 4: Resolve ticket
    ticket_id = list(manager.escalation_log.keys())[0]
    print("\nTest 4 — Resolving ticket:")
    resolve = manager.integrate_human_response(
        ticket_id,
        "Refund has been processed. Allow 3-5 business days."
    )
    print(resolve)