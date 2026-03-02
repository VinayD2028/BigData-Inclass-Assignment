#===========================================================
# Encounter_Manager.rb
#-----------------------------------------------------------
# Tracks encounters by map/area for progression and Nuzlocke.
#===========================================================

module PokemonWorldChronicles
  module EncounterManager
    #-------------------------------------------------------
    # Configuration
    #-------------------------------------------------------
    module Config
      # If true, print simple debug output to console/log.
      DEBUG_LOGGING = true
    end

    # Internal structure example:
    # {
    #   map_id_integer => {
    #     first_species: :PIKACHU,
    #     species_seen:  [:PIKACHU, :PIDGEY]
    #   }
    # }
    @encounter_data = {}

    class << self
      # Record an encounter event.
      # species_id can be a symbol or internal species identifier.
      def record_encounter(map_id, species_id)
        @encounter_data[map_id] ||= { first_species: nil, species_seen: [] }
        area_data = @encounter_data[map_id]

        area_data[:first_species] ||= species_id
        area_data[:species_seen] << species_id unless area_data[:species_seen].include?(species_id)

        log("Recorded encounter on map #{map_id}: #{species_id}")
      end

      # Returns true if the first encounter has already happened on this map.
      def first_encounter_taken?(map_id)
        area_data = @encounter_data[map_id]
        return false if area_data.nil?

        !area_data[:first_species].nil?
      end

      # Fetch all known encounter data (read-only use recommended).
      def data
        @encounter_data
      end

      # Clear all encounter data (useful for tests/debug only).
      def reset!
        @encounter_data = {}
      end

      private

      def log(message)
        return unless Config::DEBUG_LOGGING
        puts("[EncounterManager] #{message}")
      end
    end
  end
end
