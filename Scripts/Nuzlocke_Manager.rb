#===========================================================
# Nuzlocke_Manager.rb
#-----------------------------------------------------------
# Beginner-friendly template for managing Nuzlocke rules.
# Place this script in your custom scripts section.
#===========================================================

module PokemonWorldChronicles
  module NuzlockeManager
    #-------------------------------------------------------
    # Configuration
    #-------------------------------------------------------
    module Config
      # Enable or disable Nuzlocke mode by default for new saves.
      DEFAULT_ENABLED = false

      # What to do when a party member faints:
      # :release => force release
      # :box_lock => send to a locked box slot
      # :mark_only => mark as unusable, but keep available for later dev tools
      FAINT_RULE = :mark_only
    end

    #-------------------------------------------------------
    # Runtime state helpers
    #-------------------------------------------------------
    @enabled = Config::DEFAULT_ENABLED

    class << self
      # Returns true if Nuzlocke mode is active.
      def enabled?
        @enabled
      end

      # Turn Nuzlocke mode on or off.
      def set_enabled(value)
        @enabled = !!value
      end

      # Called when an encounter starts.
      # Return true if encounter is allowed, false if blocked by rules.
      def allow_encounter?(map_id)
        # TODO: Connect with EncounterManager.first_encounter_taken?(map_id)
        # Example behavior: block encounter if first encounter already happened.
        return true
      end

      # Called when a party member faints.
      # pokemon is expected to be a Pokémon object from Essentials.
      def on_pokemon_fainted(pokemon)
        # TODO: Apply Config::FAINT_RULE to this Pokémon.
        # Keep this simple while learning. Start with messages/logging.
        return if pokemon.nil?
      end

      # Basic debug summary for quick testing.
      def debug_summary
        {
          enabled: enabled?,
          faint_rule: Config::FAINT_RULE
        }
      end
    end
  end
end
