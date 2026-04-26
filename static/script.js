/**
 * script.js
 * Client-side interactivity for LoanSense AI
 */

document.addEventListener('DOMContentLoaded', () => {

  /* ── 1. Radio card visual selection ───────────────────────── */
  document.querySelectorAll('.radio-card').forEach(card => {
    const radio = card.querySelector('input[type="radio"]');
    if (!radio) return;

    // Sync initial state
    if (radio.checked) card.classList.add('selected');

    radio.addEventListener('change', () => {
      // Clear siblings in same group
      const name = radio.name;
      document.querySelectorAll(`input[name="${name}"]`).forEach(r => {
        r.closest('.radio-card')?.classList.remove('selected');
      });
      card.classList.add('selected');
    });

    // Allow clicking anywhere on the card
    card.addEventListener('click', () => {
      radio.checked = true;
      radio.dispatchEvent(new Event('change', { bubbles: true }));
    });
  });

  /* ── 2. Form validation before submit ──────────────────────── */
  const form = document.getElementById('loanForm');
  if (form) {
    form.addEventListener('submit', (e) => {
      const errors = [];

      // Required selects
      ['Gender', 'Married', 'Dependents', 'Education',
       'Self_Employed', 'Property_Area', 'Loan_Amount_Term'].forEach(name => {
        const el = form.querySelector(`[name="${name}"]`);
        if (el && !el.value) errors.push(`Please select a value for "${name.replace('_',' ')}".`);
      });

      // Required numbers
      const appIncome = parseFloat(form.querySelector('[name="ApplicantIncome"]')?.value);
      if (isNaN(appIncome) || appIncome <= 0)
        errors.push('Applicant Income must be a positive number.');

      const loanAmt = parseFloat(form.querySelector('[name="LoanAmount"]')?.value);
      if (isNaN(loanAmt) || loanAmt < 10000 || loanAmt > 999999)
        errors.push('Loan Amount must be between ₹10,000 and ₹9,99,999.');

      // Credit history radio
      const creditSelected = form.querySelector('input[name="Credit_History"]:checked');
      if (!creditSelected) errors.push('Please select your Credit History.');

      if (errors.length > 0) {
        e.preventDefault();
        showInlineErrors(errors);
        return;
      }

      // Show loading state
      const btn    = document.getElementById('submitBtn');
      const loader = document.getElementById('btnLoader');
      const text   = btn?.querySelector('.btn-text');
      if (btn && loader && text) {
        text.style.display   = 'none';
        loader.style.display = 'inline';
        btn.disabled = true;
      }
    });
  }

  /* ── 3. Inline error banner helper ─────────────────────────── */
  function showInlineErrors(errors) {
    let banner = document.getElementById('error-banner');
    if (!banner) {
      banner = document.createElement('div');
      banner.id = 'error-banner';
      banner.className = 'alert alert-error';
      const mainContent = document.querySelector('.main-content');
      mainContent?.insertBefore(banner, mainContent.firstChild);
    }
    banner.innerHTML = `
      <span class="alert-icon">⚠️</span>
      <div>
        <strong>Please fix the following errors:</strong>
        <ul class="error-list">
          ${errors.map(e => `<li>${e}</li>`).join('')}
        </ul>
      </div>`;
    banner.scrollIntoView({ behavior: 'smooth', block: 'center' });
  }

  /* ── 4. Confidence bar animation (result page) ─────────────── */
  const bar = document.querySelector('.confidence-bar');
  if (bar) {
    const target = parseFloat(bar.getAttribute('data-target') || '0');
    bar.style.width = '0%';
    setTimeout(() => { bar.style.width = target + '%'; }, 400);
  }

  /* ── 5. Result card entrance animation ─────────────────────── */
  const resultCard = document.getElementById('resultCard');
  if (resultCard) {
    requestAnimationFrame(() => {
      setTimeout(() => resultCard.classList.add('animate-in'), 50);
    });
  }

  /* ── 6. Number input formatting — clamp only after user finishes typing ── */
  document.querySelectorAll('input[type="number"]').forEach(input => {
    // Only clamp on blur so the user can type freely without values jumping
    input.addEventListener('blur', () => {
      const val = parseFloat(input.value);
      if (isNaN(val)) return;
      const min = parseFloat(input.min);
      const max = parseFloat(input.max);
      if (!isNaN(min) && val < min) input.value = min;
      if (!isNaN(max) && val > max) input.value = max;
    });
  });

  /* ── 7. Smooth scroll for anchor links ─────────────────────── */
  document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
      const target = document.querySelector(this.getAttribute('href'));
      if (target) {
        e.preventDefault();
        target.scrollIntoView({ behavior: 'smooth' });
      }
    });
  });

});
